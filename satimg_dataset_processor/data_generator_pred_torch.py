"""PyTorch Dataset for the window-addressable TS-SatFire NPZ format.

Comment legend requested for this migration:
    # * NEW: behavior added or changed for the new NPZ/DataLoader workflow.
    # - OLD: behavior retained from the original FireDataset implementation.
"""

import json
import os
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
# * NEW: Path must be imported from pathlib; `import Path` is not valid here.
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF


class Normalize:
    """Normalize selected input channels using fixed dataset statistics."""

    def __init__(self, mean, std, dont_normalize_idc):
        # - OLD: retain the original per-channel normalization statistics.
        self.mean = mean
        self.std = std
        self.dont_normalize_idc = set(dont_normalize_idc)

        # * NEW: fail early when the supplied statistics are inconsistent.
        if len(self.mean) != len(self.std):
            raise ValueError(
                "Normalization mean/std lengths differ: "
                f"{len(self.mean)} versus {len(self.std)}"
            )

    def __call__(self, sample):
        # * NEW: validate the channel dimension before modifying the tensor.
        if sample.ndim != 4:
            raise ValueError(
                "Expected input shaped (C, T, H, W), but found "
                f"{tuple(sample.shape)}"
            )

        if sample.shape[0] != len(self.mean):
            raise ValueError(
                f"Expected {len(self.mean)} channels for normalization, "
                f"but found {sample.shape[0]}"
            )

        # - OLD: normalize one channel at a time and skip degree/land-cover
        # - OLD: channels, which require special processing later.
        for channel_idx in range(len(self.mean)):
            if channel_idx not in self.dont_normalize_idc:
                sample[channel_idx] = (
                    sample[channel_idx] - self.mean[channel_idx]
                ) / self.std[channel_idx]

        return sample


# * NEW: a fire's windows are stored either as one p<id>.npz archive or as a
# * NEW: p<id>/ folder holding that archive's members unzipped (data_000.npy,
# * NEW: label_000.npy, ...). The arrays are byte-identical; reading a folder
# * NEW: skips zlib, which took about half of each window's loading time.
UNZIP_MARKER = ".source.json"


def fire_name(source):
    """Return p<id> for either a p<id>.npz archive or a p<id>/ folder."""
    source = Path(source)
    return source.stem if source.suffix == ".npz" else source.name


def find_fire_sources(split_dir):
    """List a split's fires, each as its archive or its unzipped folder.

    A folder counts once it holds num_windows.npy and is used instead of an
    archive of the same name. Hidden names (unzip work folders) are skipped.
    """
    split_dir = Path(split_dir)
    if not split_dir.is_dir():
        return []
    sources = {}
    for path in split_dir.iterdir():
        if path.name.startswith("."):
            continue
        if path.suffix == ".npz" and path.is_file():
            sources.setdefault(path.stem, path)
        elif path.is_dir() and (path / "num_windows.npy").is_file():
            sources[path.name] = path
    # Same order as sorting the archive names, so window indexes match runs
    # made before unzipped folders existed.
    return [sources[name] for name in sorted(sources, key=lambda n: f"{n}.npz")]


class _FireFolder:
    """Read an unzipped p<id>/ folder through np.load's NpzFile interface."""

    def __init__(self, folder):
        self.folder = Path(folder)

    @cached_property
    def files(self):
        return [path.stem for path in self.folder.glob("*.npy")]

    def __getitem__(self, key):
        return np.load(self.folder / f"{key}.npy", allow_pickle=False)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def open_fire(source):
    """Open a fire archive or unzipped folder for `with ... as archive`."""
    if Path(source).is_dir():
        return _FireFolder(source)
    return np.load(source, allow_pickle=False)


def _archive_signature(archive_path):
    archive_path = Path(archive_path).resolve()
    stat = archive_path.stat()
    return {
        "archive": str(archive_path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _read_unzip_marker(folder):
    try:
        return json.loads((folder / UNZIP_MARKER).read_text())
    except (OSError, ValueError):
        return None


def _unzip_fire(archive_path, folder):
    """Unzip one archive into a hidden work folder, then rename it to folder."""
    work = folder.with_name(f".{folder.name}.partial")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            name = member.filename
            if (
                not name.endswith(".npy")
                or name.startswith(".")
                or "/" in name
                or "\\" in name
            ):
                raise ValueError(f"{archive_path}: unexpected member {name!r}")
            # ZipFile checks each member's CRC as it is read.
            with archive.open(member) as src, open(work / name, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
    (work / UNZIP_MARKER).write_text(json.dumps(_archive_signature(archive_path)))
    if folder.exists():
        shutil.rmtree(folder)
    work.rename(folder)


def unzip_data_root(
    data_root,
    unzip_root,
    splits=("train", "val", "test"),
    workers=None,
):
    """Unzip each split's archives once into unzip_root; return the root to load.

    Returns unzip_root once every data_root/<split>/p<id>.npz has a matching
    unzip_root/<split>/p<id>/ folder. Folders already unzipped from the same
    archive (same path, size and modification time) are reused, so this is
    cheap after the first call. Returns data_root unchanged, and FireDataset
    reads the archives directly, when data_root holds no archives (it is
    already unzipped) or when unzip_root is not writable or lacks the space.
    """
    data_root, unzip_root = Path(data_root), Path(unzip_root)
    sources = {split: find_fire_sources(data_root / split) for split in splits}
    if not any(sources.values()):
        return data_root
    if any(path.suffix != ".npz" for paths in sources.values() for path in paths):
        print(f"{data_root} already holds unzipped fire folders; using it as is")
        return data_root

    jobs = []
    stale = []
    for split, archives in sources.items():
        split_dir = unzip_root / split
        for archive in archives:
            folder = split_dir / archive.stem
            if _read_unzip_marker(folder) != _archive_signature(archive):
                jobs.append((archive, folder))
        # Folders this function made for fires that data_root no longer has.
        names = {archive.stem for archive in archives}
        if split_dir.is_dir():
            stale += [
                folder
                for folder in split_dir.iterdir()
                if folder.name not in names and (folder / UNZIP_MARKER).is_file()
            ]

    for folder in stale:
        shutil.rmtree(folder)
    if not jobs:
        print(f"Using unzipped windows in {unzip_root}")
        return unzip_root

    needed = 0
    for archive, _ in jobs:
        with zipfile.ZipFile(archive) as zipped:
            needed += sum(member.file_size for member in zipped.infolist())
    try:
        unzip_root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(unzip_root).free
    except OSError as error:
        print(f"Cannot write {unzip_root} ({error}); reading the archives directly")
        return data_root
    if needed + (1 << 30) > free:
        print(
            f"Unzipping needs {needed / 2**30:.1f} GiB but {unzip_root} has "
            f"{free / 2**30:.1f} GiB free; reading the archives directly"
        )
        return data_root

    print(
        f"Unzipping {len(jobs)} fire archives ({needed / 2**30:.1f} GiB) "
        f"into {unzip_root}"
    )
    try:
        # zlib and file writes release the GIL, so threads unzip in parallel.
        with ThreadPoolExecutor(max_workers=workers or os.cpu_count()) as pool:
            for _ in pool.map(lambda job: _unzip_fire(*job), jobs):
                pass
    except OSError as error:
        print(f"Unzipping failed ({error}); reading the archives directly")
        return data_root
    return unzip_root


# * NEW: TESSERA files hold 128 embedding bands; one validity mask is added.
TESSERA_BANDS = 128
TESSERA_CHANNELS = TESSERA_BANDS + 1

# * NEW: stored (embedding, coverage) arrays by resolved TESSERA path, so each
# * NEW: file is unzipped once per process (about 16 MiB per fire as float16)
# * NEW: and shared by compute_tessera_stats and every FireDataset.
_TESSERA_CACHE = {}


def read_tessera(path):
    """Return one TESSERA file's stored (embedding, coverage), read once."""
    path = Path(path).resolve()
    if path not in _TESSERA_CACHE:
        with np.load(path, allow_pickle=False) as archive:
            embedding = archive["embedding"]
            coverage = archive["coverage"]
        if embedding.shape != (TESSERA_BANDS, 256, 256):
            raise ValueError(
                f"{path}: embedding has shape {embedding.shape}; "
                f"expected ({TESSERA_BANDS}, 256, 256)"
            )
        # Shared by every dataset and DataLoader worker: never modify them.
        embedding.flags.writeable = False
        coverage.flags.writeable = False
        _TESSERA_CACHE[path] = (embedding, coverage)
    return _TESSERA_CACHE[path]


def compute_tessera_stats(tessera_paths, min_coverage=0.5):
    """Per-band mean/std over well-covered pixels of the given TESSERA files.

    Compute this from the training fires only and reuse it for validation and
    test, so no statistics leak from held-out fires.
    """
    total = np.zeros(TESSERA_BANDS, dtype=np.float64)
    total_sq = np.zeros(TESSERA_BANDS, dtype=np.float64)
    count = 0
    for path in tessera_paths:
        embedding, coverage = read_tessera(path)
        embedding = embedding.astype(np.float64)
        valid = coverage.astype(np.float32) >= min_coverage
        values = embedding[:, valid]
        total += values.sum(axis=1)
        total_sq += (values ** 2).sum(axis=1)
        count += values.shape[1]
    if count == 0:
        raise RuntimeError("No TESSERA pixels with sufficient coverage")
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean ** 2, 1e-12))
    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "min_coverage": float(min_coverage),
    }


class FireDataset(Dataset):
    """Expose every temporal window across every fire as one Dataset item.

    Expected archive layout::

        fire.npz
        ├── format_version   scalar, currently 2
        ├── num_windows      scalar
        ├── data_000         (C, T, H, W)
        ├── label_000        (H, W) or (T, H, W)
        ├── data_001
        └── label_001

    Because each window is a separate ZIP member, __getitem__ decompresses
    only the requested data/label pair rather than the entire fire.

    A fire may instead be a p<id>/ folder of the same members as plain .npy
    files (see unzip_data_root); the format is detected per fire.

    Each item is {"data": (43, T, H, W), "labels": (2, H, W)}, plus
    "static": (129, H, W) in TESSERA mode. The trainer repeats "static" across
    the T days on the GPU and appends it after the 43 channels.
    """

    def __init__(
        self,
        npz_dir,
        ts_length,
        use_augmentations=False,
        n_channel=27,
        label_sel=0,
        target_is_single_day=False,
        tessera_dir=None,
        tessera_stats=None,
    ):
        # * NEW: data and labels now live together in version-2 NPZ files,
        # * NEW: or in unzipped p<id>/ folders of the same members.
        self.npz_dir = Path(npz_dir)
        self.npz_files = find_fire_sources(self.npz_dir)

        if not self.npz_files:
            raise RuntimeError(
                f"No .npz archives or unzipped fire folders found in {self.npz_dir}"
            )

        # * NEW: optional TESSERA mode. Each fire archive p<id>.npz pairs with
        # * NEW: <tessera_dir>/p<id>.npz; None keeps the 27-channel behavior.
        self.tessera_dir = Path(tessera_dir) if tessera_dir is not None else None
        self.tessera_stats = tessera_stats
        # * NEW: each fire's TESSERA map is read here, once, rather than in
        # * NEW: every __getitem__. DataLoader workers forked later (Linux,
        # * NEW: including Kaggle) share these arrays instead of copying them.
        self.tessera_maps = {}
        if self.tessera_dir is not None:
            if tessera_stats is None:
                raise ValueError("tessera_stats is required with tessera_dir")
            missing = [
                f"{fire_name(path)}.npz"
                for path in self.npz_files
                if not (self.tessera_dir / f"{fire_name(path)}.npz").exists()
            ]
            if missing:
                raise RuntimeError(
                    f"{len(missing)} archives in {self.npz_dir} have no TESSERA "
                    f"file in {self.tessera_dir}: {missing[:10]}"
                )
            for path in self.npz_files:
                name = fire_name(path)
                embedding, coverage = read_tessera(self.tessera_dir / f"{name}.npz")
                valid = (
                    coverage.astype(np.float32) >= tessera_stats["min_coverage"]
                )
                self.tessera_maps[name] = (embedding, valid)

        # - OLD: retain these options so existing training configuration and
        # - OLD: FireDataset calls continue to express the same behavior.
        self.n_channel = n_channel
        self.label_sel = label_sel
        self.ts_length = ts_length
        self.target_is_single_day = target_is_single_day
        self.use_augmentations = use_augmentations

        # - OLD: these channels contain angles in degrees and are transformed
        # - OLD: with sine after geometric augmentation.
        self.indices_of_degree_features = [12, 18, 24]

        # * NEW: a global index lets PyTorch shuffle windows across all fires.
        # * NEW: each entry is `(archive_path, window_index_inside_archive)`.
        self.index = []
        for npz_path in self.npz_files:
            with open_fire(npz_path) as archive:
                # * NEW: reject old archives rather than silently loading the
                # * NEW: whole `data` member and defeating lazy decompression.
                if "num_windows" not in archive.files:
                    raise RuntimeError(
                        f"{npz_path} uses the old NPZ format. Expected "
                        "num_windows plus data_###/label_### members."
                    )

                number_of_windows = int(
                    np.asarray(archive["num_windows"]).item()
                )

                if "format_version" in archive.files:
                    format_version = int(
                        np.asarray(archive["format_version"]).item()
                    )
                    if format_version != 2:
                        raise RuntimeError(
                            f"{npz_path} has unsupported format version "
                            f"{format_version}; expected version 2"
                        )

                # * NEW: checking member names reads only the ZIP directory;
                # * NEW: it does not decompress the large window arrays.
                for window_idx in range(number_of_windows):
                    suffix = f"{window_idx:03d}"
                    data_key = f"data_{suffix}"
                    label_key = f"label_{suffix}"

                    if data_key not in archive.files:
                        raise RuntimeError(
                            f"{npz_path} is missing member {data_key}"
                        )
                    if label_key not in archive.files:
                        raise RuntimeError(
                            f"{npz_path} is missing member {label_key}"
                        )

                    self.index.append((npz_path, window_idx))

        if not self.index:
            raise RuntimeError(
                f"NPZ files in {self.npz_dir} contain no temporal windows"
            )

        # - OLD: retain the original 27-channel training statistics.
        self.normalizer = Normalize(
            mean=[
                18.224253,
                26.95519,
                20.09066,
                318.25967,
                308.78717,
                14.165086,
                291.29214,
                288.97382,
                5110.5547,
                2556.2627,
                0.3907487,
                3.4994626,
                216.23518,
                276.5463,
                291.8275,
                70.32086,
                0.0054306216,
                10.120554,
                175.33012,
                1290.8367,
                -1.5219007,
                7.3989105,
                7.584937,
                1.4395763,
                3.306973,
                19.259102,
                0.0057929577,
            ],
            std=[
                15.438321,
                14.408274,
                10.552524,
                13.1312475,
                12.155249,
                9.652911,
                12.435288,
                8.750125,
                2400.766,
                1206.8983,
                2.37979,
                1.6343528,
                85.730644,
                47.332256,
                50.045837,
                22.48386,
                0.0021515382,
                8.429097,
                104.73222,
                823.01483,
                1.9954495,
                4.1257873,
                26.547232,
                1.2017097,
                48.207355,
                5.4114914,
                0.0017134654,
            ],
            # - OLD: do not normalize angle channels or land-cover classes.
            dont_normalize_idc=self.indices_of_degree_features + [21],
        )

        # - OLD: land-cover values 1..17 are converted to 17 one-hot channels.
        # * NEW: a leading zero row encodes missing land cover (0) as no class.
        self.one_hot_matrix = torch.cat(
            (torch.zeros(1, 17), torch.eye(17)), dim=0
        )

    def __len__(self):
        # * NEW: the Dataset length is the number of windows across all fires,
        # * NEW: not the number of NPZ files and not an old `num_samples` field.
        return len(self.index)

    def __getitem__(self, idx):
        # * NEW: DataLoader supplies a global window index. load_data maps it
        # * NEW: to exactly one member pair inside exactly one fire archive.
        x, y = self.load_data(idx)

        # - OLD: normalize before augmentation. The degree and land-cover
        # - OLD: channels are excluded from this normalization operation.
        x = self.normalizer(x)

        # * NEW: the fire's TESSERA map is static, so it is returned once as
        # * NEW: (129, H, W). Repeating it across the T days here cost T times
        # * NEW: its size in host RAM; the trainer repeats it on the GPU.
        static = self.load_tessera(idx) if self.tessera_dir is not None else None

        # - OLD: augment training examples but leave validation unchanged.
        # * NEW: the TESSERA map gets the same flips and rotation.
        if self.use_augmentations:
            x, y, static = self.augment(x, y, static)

        # - OLD: convert degrees to sine values and one-hot encode land cover.
        x = self.preprocess(x)

        # - OLD: retain the dictionary interface used by the training loop.
        sample = {
            "data": x,
            "labels": y,
        }
        if static is not None:
            sample["static"] = static
        return sample

    def load_data(self, idx):
        """Load and decode exactly one temporal window.

        This is lazy with respect to disk: opening the archive reads its ZIP
        index, and accessing data_###/label_### decompresses only those two
        members into RAM.
        """

        # * NEW: resolve the global Dataset index into fire and window indexes.
        npz_path, window_idx = self.index[idx]
        suffix = f"{window_idx:03d}"
        data_key = f"data_{suffix}"
        label_key = f"label_{suffix}"

        # * NEW: only one approximately 67.5-MiB input window is decompressed,
        # * NEW: rather than every window stored for this fire. An unzipped
        # * NEW: folder reads the same two members without decompressing.
        with open_fire(npz_path) as archive:
            data_array = archive[data_key]
            label_array = archive[label_key]

        # * NEW: NPZ-loaded arrays normally own writable memory already. Copy
        # * NEW: only if necessary because normalization modifies x in place.
        if not data_array.flags.writeable:
            data_array = data_array.copy()

        # * NEW: do not use squeeze(); it could accidentally remove a real
        # * NEW: temporal dimension when T == 1.
        x = torch.from_numpy(data_array).to(dtype=torch.float32)
        label = torch.from_numpy(label_array)

        # * NEW: validate the stored input instead of allowing a later model
        # * NEW: layer to fail with a less informative shape error.
        expected_x_shape = (
            self.n_channel,
            self.ts_length,
            256,
            256,
        )
        if tuple(x.shape) != expected_x_shape:
            raise ValueError(
                f"{npz_path}:{data_key} has shape {tuple(x.shape)}; "
                f"expected {expected_x_shape}"
            )

        # - OLD: target_is_single_day controls whether each target is one map
        # - OLD: or an entire temporal sequence of maps.
        if self.target_is_single_day:
            # * NEW: accept a harmless leading singleton left by older exports.
            if label.ndim == 3 and label.shape[0] == 1:
                label = label[0]

            if label.ndim != 2:
                raise ValueError(
                    f"target_is_single_day=True requires (H, W), but "
                    f"{npz_path}:{label_key} has shape {tuple(label.shape)}"
                )
        else:
            if label.ndim != 3:
                raise ValueError(
                    f"target_is_single_day=False requires (T, H, W), but "
                    f"{npz_path}:{label_key} has shape {tuple(label.shape)}"
                )

            if label.shape[0] != self.ts_length:
                raise ValueError(
                    f"{npz_path}:{label_key} contains {label.shape[0]} "
                    f"target frames; expected {self.ts_length}"
                )

        if tuple(label.shape[-2:]) != (256, 256):
            raise ValueError(
                f"{npz_path}:{label_key} has spatial shape "
                f"{tuple(label.shape[-2:])}; expected (256, 256)"
            )

        # - OLD: represent the binary target as two channels:
        # - OLD: channel 0 is background and channel 1 is positive/fire.
        # * NEW: stack infers either (2,H,W) or (2,T,H,W), avoiding separate
        # * NEW: hard-coded allocation branches.
        y = torch.stack(
            (
                label == 0,
                label > 0,
            ),
            dim=0,
        ).long()

        return x, y

    def load_tessera(self, idx):
        """Return (129, H, W): normalized embedding plus validity mask.

        Pixels whose TESSERA coverage is below the training threshold are
        zeroed (the normalized mean) and marked 0 in the mask channel.
        """
        npz_path, _ = self.index[idx]
        # * NEW: the stored map was read in __init__; only normalization,
        # * NEW: unchanged, runs per window.
        embedding, valid = self.tessera_maps[fire_name(npz_path)]
        embedding = embedding.astype(np.float32)

        mean = self.tessera_stats["mean"][:, None, None]
        std = self.tessera_stats["std"][:, None, None]
        embedding = np.where(valid, (embedding - mean) / std, 0.0)

        static = np.concatenate(
            (embedding, valid[None].astype(np.float32)),
            axis=0,
        ).astype(np.float32)
        return torch.from_numpy(static)

    def preprocess(self, x):
        # - OLD: transform angular degree features after augmentation.
        x[self.indices_of_degree_features] = torch.sin(
            torch.deg2rad(x[self.indices_of_degree_features])
        )

        # - OLD: replace the integer land-cover channel with 17 one-hot planes.
        new_shape = (
            x.shape[1],
            x.shape[2],
            x.shape[3],
            self.one_hot_matrix.shape[1],
        )

        # - OLD: source land-cover classes use integer identifiers 1..17.
        landcover_classes = x[21].long()

        # * NEW: 0 is a missing value (preprocessing replaces NaN with 0). The
        # * NEW: paper reads a zero land cover as "an absence of all land
        # * NEW: covers", so it gets an all-zero encoding; the released code
        # * NEW: indexed it as -1, silently misclassifying it as class 17.
        invalid_landcover = (landcover_classes < 0) | (
            landcover_classes > self.one_hot_matrix.shape[1]
        )
        if torch.any(invalid_landcover):
            invalid_values = torch.unique(
                landcover_classes[invalid_landcover]
            ).tolist()
            raise ValueError(
                "Land-cover channel contains values outside 0..17: "
                f"{invalid_values[:20]}"
            )

        # Row 0 of the padded matrix is the all-zero encoding for class 0.
        landcover_flat = landcover_classes.flatten()
        landcover_encoding = self.one_hot_matrix[landcover_flat].reshape(
            new_shape
        ).permute(3, 0, 1, 2)

        x = torch.cat(
            [x[:21], landcover_encoding, x[22:]],
            dim=0,
        )
        return x

    def augment(self, x, y, static=None):
        # - OLD: independently choose horizontal flip, vertical flip, and a
        # - OLD: multiple-of-90-degree rotation for each training sample.
        # * NEW: `static` (the TESSERA map, or None) is transformed with x.
        hflip = bool(np.random.random() > 0.5)
        vflip = bool(np.random.random() > 0.5)
        rotate = int(np.floor(np.random.random() * 4))

        if hflip:
            x = TF.hflip(x)
            y = TF.hflip(y)
            if static is not None:
                static = TF.hflip(static)

            # - OLD: reflect directional angles across the vertical axis.
            x[self.indices_of_degree_features] = (
                360 - x[self.indices_of_degree_features]
            ) % 360

        if vflip:
            x = TF.vflip(x)
            y = TF.vflip(y)
            if static is not None:
                static = TF.vflip(static)

            # - OLD: reflect directional angles across the horizontal axis.
            x[self.indices_of_degree_features] = (
                180 - x[self.indices_of_degree_features]
            ) % 360

        if rotate:
            angle = rotate * 90

            # * NEW: torchvision rotates the last two dimensions and supports
            # * NEW: both (2,H,W) and (2,T,H,W); no squeeze/unsqueeze is needed.
            x = TF.rotate(x, angle)
            y = TF.rotate(y, angle)
            if static is not None:
                static = TF.rotate(static, angle)

            # - OLD: rotate directional values consistently with the raster.
            x[self.indices_of_degree_features] = (
                x[self.indices_of_degree_features] - angle
            ) % 360

        return x, y, static


if __name__ == "__main__":
    # * NEW: both data and labels are stored in the same NPZ directory.
    npz_dir = "/content/data/train"

    # * NEW: this example matches the generated prediction archives:
    # * NEW: 27 channels, ten input days, and one next-day target map.
    train_dataset = FireDataset(
        npz_dir=npz_dir,
        ts_length=10,
        n_channel=27,
        target_is_single_day=True,
        use_augmentations=True,
    )

    # - OLD: this remains an ordinary map-style DataLoader. shuffle=True now
    # * NEW: shuffles every window globally across all fire archives.
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        # * NEW: start with zero workers in Colab to establish RAM usage.
        num_workers=0,
        pin_memory=False,
    )

    print(f"Total batches: {len(train_dataloader)}")

    for batch_number, batch in enumerate(train_dataloader, start=1):
        data_batch = batch["data"]
        labels_batch = batch["labels"]

        print(
            f"Batch {batch_number}/{len(train_dataloader)} | "
            f"data: {tuple(data_batch.shape)} | "
            f"labels: {tuple(labels_batch.shape)}"
        )
