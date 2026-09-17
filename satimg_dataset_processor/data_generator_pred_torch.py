"""PyTorch Dataset for the window-addressable TS-SatFire NPZ format.

Comment legend requested for this migration:
    # * NEW: behavior added or changed for the new NPZ/DataLoader workflow.
    # - OLD: behavior retained from the original FireDataset implementation.
"""

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
    """

    def __init__(
        self,
        npz_dir,
        ts_length,
        use_augmentations=False,
        n_channel=27,
        label_sel=0,
        target_is_single_day=False,
    ):
        # * NEW: data and labels now live together in version-2 NPZ files.
        self.npz_dir = Path(npz_dir)
        self.npz_files = sorted(self.npz_dir.glob("*.npz"))

        if not self.npz_files:
            raise RuntimeError(f"No .npz files found in {self.npz_dir}")

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
            with np.load(npz_path, allow_pickle=False) as archive:
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
        self.one_hot_matrix = torch.eye(17, dtype=torch.float32)

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

        # - OLD: augment training examples but leave validation unchanged.
        if self.use_augmentations:
            x, y = self.augment(x, y)

        # - OLD: convert degrees to sine values and one-hot encode land cover.
        x = self.preprocess(x)

        # - OLD: retain the dictionary interface used by the training loop.
        return {
            "data": x,
            "labels": y,
        }

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
        # * NEW: rather than every window stored for this fire.
        with np.load(npz_path, allow_pickle=False) as archive:
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
            self.one_hot_matrix.shape[0],
        )

        # - OLD: source land-cover classes use integer identifiers 1..17.
        landcover_classes = x[21].long()

        # * NEW: reject invalid identifiers rather than allowing zero to become
        # * NEW: index -1, which would silently misclassify it as class 17.
        invalid_landcover = (landcover_classes < 1) | (
            landcover_classes > self.one_hot_matrix.shape[0]
        )
        if torch.any(invalid_landcover):
            invalid_values = torch.unique(
                landcover_classes[invalid_landcover]
            ).tolist()
            raise ValueError(
                "Land-cover channel contains values outside 1..17: "
                f"{invalid_values[:20]}"
            )

        landcover_flat = landcover_classes.flatten() - 1
        landcover_encoding = self.one_hot_matrix[landcover_flat].reshape(
            new_shape
        ).permute(3, 0, 1, 2)

        x = torch.cat(
            [x[:21], landcover_encoding, x[22:]],
            dim=0,
        )
        return x

    def augment(self, x, y):
        # - OLD: independently choose horizontal flip, vertical flip, and a
        # - OLD: multiple-of-90-degree rotation for each training sample.
        hflip = bool(np.random.random() > 0.5)
        vflip = bool(np.random.random() > 0.5)
        rotate = int(np.floor(np.random.random() * 4))

        if hflip:
            x = TF.hflip(x)
            y = TF.hflip(y)

            # - OLD: reflect directional angles across the vertical axis.
            x[self.indices_of_degree_features] = (
                360 - x[self.indices_of_degree_features]
            ) % 360

        if vflip:
            x = TF.vflip(x)
            y = TF.vflip(y)

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

            # - OLD: rotate directional values consistently with the raster.
            x[self.indices_of_degree_features] = (
                x[self.indices_of_degree_features] - angle
            ) % 360

        return x, y


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

    batch = next(iter(train_dataloader))
    print("data batch:", batch["data"].shape)
    print("label batch:", batch["labels"].shape)
