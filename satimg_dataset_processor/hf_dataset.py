"""Hugging Face pipeline for the TS-SatFire next-day prediction dataset.

Ported from the SwinUNETR_data_grab Colab notebook. Repositories, tokens, and
working directories are arguments rather than notebook globals, so a Colab
notebook only has to read its secrets and call these functions.

Settings (window length/stride, repos, split) live in
satimg_dataset_processor/config.py. Hugging Face dataset repos:
    SRC_REPO        ts-satfire                   raw GeoTIFFs + ROI CSVs
    PROCESSED_REPO  ts-satfire-processed         old-format NPZ (data/labels) in data.zip
    WINDOW_REPO     ts-satfire-processed-window  windowed NPZ per fire (training data);
                    one repo per (TS_LENGTH, TS_INTERVAL), see config.window_repo_for
    TESSERA_REPO    ts-tesserafire               prior-year TESSERA per fire

Typical flow:
    process_data(...)            SRC_REPO -> WINDOW_REPO {train,val,test}/
    tessera.process_tessera(...) SRC_REPO + TESSERA -> TESSERA_REPO tessera/
    download_processed_data(...) WINDOW_REPO [+ TESSERA_REPO] -> <data>/{train,val,test}[,tessera]
    upload_data_zip(...)         <data> -> data.zip in WINDOW_REPO
    tessera.upload_tessera_zip() TESSERA_REPO tessera/ -> tessera.zip in TESSERA_REPO
    pull_zip_data_to(...)        a repo's zip -> <data>
    tessera.pull_tessera_data()  tessera.zip -> <data>/tessera, drops 2017 windows
    pull_processed_data(...)     PROCESSED_REPO data.zip -> <data> (old format)
    convert_and_upload(...)      old-format NPZ under <data> -> WINDOW_REPO
"""

import gc
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download, login
from tqdm.auto import tqdm

from satimg_dataset_processor.config import (
    DEFAULT_LABEL_SEL,
    LABEL_SEL_FILE,
    LEGACY_WINDOW_SETTINGS,
    PROCESSED_REPO,
    SRC_REPO,
    TESSERA_REPO,
    TEST_YEARS,
    TRAIN_VAL_YEARS,
    TS_INTERVAL,
    TS_LENGTH,
    VAL_IDS,
    WINDOW_CONFIG_FILE,
    WINDOW_REPO,
    window_repo_for,
)
from satimg_dataset_processor.satimg_dataset_processor import PredDatasetProcessor


DEST_REPO = WINDOW_REPO  # earlier name for WINDOW_REPO, kept for existing notebooks
SPLITS = ("train", "val", "test")


def list_fire_files(api, src_repo):
    """Scan source repository and group GeoTIFFs by fire and mode."""
    all_files = api.list_repo_files(
        src_repo,
        repo_type="dataset",
    )
    locations = {}

    for file_name in all_files:
        if not file_name.endswith(".tif"):
            continue

        parts = file_name.split("/")
        if len(parts) < 3:
            continue

        location = parts[-3]
        mode = parts[-2]
        locations.setdefault(location, {}).setdefault(mode, []).append(
            file_name
        )

    for location in locations:
        for mode in locations[location]:
            locations[location][mode].sort()

    return locations


# ----- DATA GRAB ----- #

class DatasetStreamer(PredDatasetProcessor):
    """Stream GeoTIFFs into one window-addressable NPZ per fire."""

    def __init__(self, src_repo, dest_repo, token, batch_size=10, work_dir="."):
        self.src_repo = src_repo
        self.dest_repo = dest_repo
        self.token = token
        self.batch_size = batch_size
        self.api = HfApi(token=token)
        login(token)

        self.temp_dir = os.path.join(work_dir, "temp")
        os.makedirs(self.temp_dir, exist_ok=True)

        self.staging_dir = os.path.join(work_dir, "staging")
        os.makedirs(self.staging_dir, exist_ok=True)

        self.api.create_repo(
            repo_id=self.dest_repo,
            repo_type="dataset",
            exist_ok=True,
        )

    def get_repo_locations(self):
        """Scan source repository and group GeoTIFFs by fire and mode."""
        return list_fire_files(self.api, self.src_repo)

    def run_split(
        self,
        target_ids,
        split_name,
        length=TS_LENGTH,
        interval=TS_INTERVAL,
        label_sel_by_id=None,
        overwrite=False,
        debug=False,
        reprocess=(),
    ):
        """Process `target_ids` into <split_name>/p<id>.npz in the dest repo.

        Fires already in the repo are skipped unless `overwrite` is set or
        they are listed in `reprocess`. Returns {repo path: label_sel} for
        every archive written.
        """
        reprocess = {str(fire_id) for fire_id in reprocess}
        written = {}
        locations = self.get_repo_locations()
        existing_files = set(
            self.api.list_repo_files(
                self.dest_repo,
                repo_type="dataset",
            )
        )

        current_staging_dir = os.path.join(
            self.staging_dir,
            split_name,
        )
        os.makedirs(current_staging_dir, exist_ok=True)
        staged_count = 0

        valid_locations = [
            location
            for location in target_ids
            if location in locations
        ]
        print(
            f"Found {len(valid_locations)} valid locations that match."
        )

        for loc_idx, loc_name in enumerate(valid_locations):
            dest_filename = f"{split_name}/p{loc_name}.npz"

            if dest_filename in existing_files and not overwrite and loc_name not in reprocess:
                print(f"Skipping {loc_name}: already processed")
                continue

            print(
                f"[{split_name.upper()} | {loc_idx + 1}/"
                f"{len(valid_locations)}] Processing: {loc_name}"
            )

            os.makedirs(self.temp_dir, exist_ok=True)
            local_map = {}
            for paths in locations[loc_name].values():
                for path in paths:
                    local_map[path] = hf_hub_download(
                        repo_id=self.src_repo,
                        filename=path,
                        repo_type="dataset",
                        cache_dir=self.temp_dir,
                        token=self.token,
                    )

            label_sel = DEFAULT_LABEL_SEL
            if label_sel_by_id is not None:
                label_sel = int(label_sel_by_id.get(loc_name, DEFAULT_LABEL_SEL))

            loc_x, loc_y = self.process_location(
                loc_name,
                local_map,
                length=length,
                interval=interval,
                label_sel=label_sel,
                debug=debug,
            )

            if loc_x is not None and len(loc_x) > 0:
                save_path = os.path.join(
                    current_staging_dir,
                    f"p{loc_name}.npz",
                )
                self._save_windowed_npz(
                    save_path,
                    data=loc_x,
                    labels=loc_y,
                    label_sel=label_sel,
                )
                written[dest_filename] = label_sel
                staged_count += 1
            else:
                print(f"{loc_name}: no output saved")

            shutil.rmtree(self.temp_dir, ignore_errors=True)
            gc.collect()

            if staged_count >= self.batch_size:
                print(
                    f"Uploading batch of {staged_count} fires to "
                    "Hugging Face"
                )
                self.api.upload_folder(
                    folder_path=current_staging_dir,
                    repo_id=self.dest_repo,
                    repo_type="dataset",
                    path_in_repo=split_name,
                )
                shutil.rmtree(
                    current_staging_dir,
                    ignore_errors=True,
                )
                os.makedirs(current_staging_dir, exist_ok=True)
                staged_count = 0

        if staged_count > 0:
            print(
                f"Uploading final batch of {staged_count} fires to "
                "Hugging Face"
            )
            self.api.upload_folder(
                folder_path=current_staging_dir,
                repo_id=self.dest_repo,
                repo_type="dataset",
                path_in_repo=split_name,
            )
            shutil.rmtree(
                current_staging_dir,
                ignore_errors=True,
            )

        return written


# ----- SPLITTING ----- #

def download_roi_csvs(src_repo=SRC_REPO, token=None, roi_dir="hf_roi"):
    """Download every ROI CSV (fire metadata) from `src_repo` into `roi_dir`."""
    os.makedirs(roi_dir, exist_ok=True)

    # obtains all roi files (metadata that maps onto fires)
    # need to pull them to get a dataframe of ids for training.
    api = HfApi(token=token)
    files = []
    for file_name in api.list_repo_files(src_repo, repo_type="dataset"):
        if file_name.endswith(".csv"):
            files.append(file_name)

    local_paths = []
    for file_name in files:
        local_path = hf_hub_download(
            filename=file_name,
            repo_id=src_repo,
            repo_type="dataset",
            local_dir=roi_dir,
            token=token,
        )
        local_paths.append(local_path)

    return local_paths


def fire_label_sel(roi_paths):
    """Map fire Id -> label_sel from every ROI CSV that has the column."""
    label_sel = {}
    for path in roi_paths:
        df = pd.read_csv(path, dtype={"Id": str})
        if "Id" not in df.columns or "label_sel" not in df.columns:
            continue
        for fire_id, value in zip(df["Id"], df["label_sel"]):
            if pd.isna(value):
                continue
            fire_id, value = str(fire_id), int(value)
            if label_sel.setdefault(fire_id, value) != value:
                raise ValueError(f"ROI CSVs disagree on label_sel for {fire_id}")
    return label_sel


def read_label_sel_manifest(api, window_repo, token=None):
    """Return {"<split>/p<id>.npz": label_sel} recorded in `window_repo`."""
    if LABEL_SEL_FILE not in api.list_repo_files(window_repo, repo_type="dataset"):
        return {}
    path = hf_hub_download(
        repo_id=window_repo,
        filename=LABEL_SEL_FILE,
        repo_type="dataset",
        token=token,
        local_dir=tempfile.mkdtemp(prefix="label_sel_"),
    )
    with open(path) as handle:
        return {key: int(value) for key, value in json.load(handle).items()}


def obtain_ids(src_repo=SRC_REPO, token=None, roi_dir="hf_roi"):
    """Download the ROI CSVs from `src_repo` and split fire IDs.

    Train/val come from 2017-2020 fires (val is the fixed VAL_IDS list);
    test comes from 2021 fires.
    """
    local_paths = download_roi_csvs(src_repo, token, roi_dir)

    # --- training and validation dataframe ---
    dfs = []
    for path in local_paths:
        print(path)
        if any(year in os.path.basename(path) for year in TRAIN_VAL_YEARS):
            dfs.append(pd.read_csv(path, dtype={'Id': str}))

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(by=['Id'])
    df['Id'] = df['Id'].astype(str)

    train_df = df[~df.Id.isin(VAL_IDS)]
    val_df = df[df.Id.isin(VAL_IDS)]

    train_ids = train_df['Id'].values.astype(str)
    val_ids = val_df['Id'].values.astype(str)

    # --- testing dataframe ---
    dfs_test = []
    for path in local_paths:
        print(path)
        if any(year in os.path.basename(path) for year in TEST_YEARS):
            dfs_test.append(pd.read_csv(path, dtype={'Id': str}))

    df_test = pd.concat(dfs_test, ignore_index=True)
    test_ids = df_test['Id'].values.astype(str)

    # validate that the vars are initialized
    print(f"Train: {len(train_ids)}")
    print(f"Val: {len(val_ids)}")
    print(f"Test: {len(test_ids)}")

    return train_ids, val_ids, test_ids


def read_window_config(api, window_repo, token=None):
    """Return {"ts_length", "ts_interval"} of `window_repo`, or None if unknown.

    Repos created before window_config.json existed hold the original
    LEGACY_WINDOW_SETTINGS windows.
    """
    files = set(api.list_repo_files(window_repo, repo_type="dataset"))
    if WINDOW_CONFIG_FILE in files:
        path = hf_hub_download(
            repo_id=window_repo,
            filename=WINDOW_CONFIG_FILE,
            repo_type="dataset",
            token=token,
            local_dir=tempfile.mkdtemp(prefix="window_config_"),
        )
        with open(path) as handle:
            return json.load(handle)
    if any(file_path.split("/")[0] in SPLITS for file_path in files):
        ts_length, ts_interval = LEGACY_WINDOW_SETTINGS
        return {"ts_length": ts_length, "ts_interval": ts_interval}
    return None


def ensure_window_config(api, window_repo, length, interval, token=None):
    """Record the window settings in `window_repo`, refusing to mix settings."""
    wanted = {"ts_length": int(length), "ts_interval": int(interval)}
    existing = read_window_config(api, window_repo, token)
    if existing is not None and existing != wanted:
        raise ValueError(
            f"{window_repo} holds windows with {existing}, not {wanted}. Use "
            f"window_repo_for({length}, {interval}) = "
            f"{window_repo_for(length, interval)!r} instead."
        )
    if WINDOW_CONFIG_FILE not in api.list_repo_files(window_repo, repo_type="dataset"):
        api.upload_file(
            path_or_fileobj=json.dumps(wanted, indent=2).encode(),
            path_in_repo=WINDOW_CONFIG_FILE,
            repo_id=window_repo,
            repo_type="dataset",
        )


def process_data(
    token,
    src_repo=SRC_REPO,
    window_repo=None,
    batch_size=10,
    length=TS_LENGTH,
    interval=TS_INTERVAL,
    splits=("train", "val", "test"),
    roi_dir="hf_roi",
    work_dir=".",
    overwrite=False,
    debug=False,
    use_roi_label_sel=True,
):
    """Stream raw fires from `src_repo` into windows of `length` input days.

    A new window starts every `interval` days. Windows go to `window_repo`,
    by default window_repo_for(length, interval), so different settings never
    share a repo. `token` needs write access to `window_repo`.

    Targets use each fire's ROI label_sel (DEFAULT_LABEL_SEL when absent),
    like the original TS-SatFire generator; `use_roi_label_sel=False` uses
    DEFAULT_LABEL_SEL for every fire. Fires already in `window_repo` are
    skipped unless `overwrite` is set, except fires whose recorded label_sel
    differs from the one wanted: those are rebuilt so every archive follows
    the current rule. The rule per archive is recorded in label_sel.json.
    """
    if window_repo is None:
        window_repo = window_repo_for(length, interval)
    print(f"Windows: {length} input days every {interval} days -> {window_repo}")

    processor = DatasetStreamer(
        src_repo=src_repo,
        dest_repo=window_repo,
        token=token,
        batch_size=batch_size,
        work_dir=work_dir,
    )
    ensure_window_config(processor.api, window_repo, length, interval, token)

    # using id metadata from above, split into appropriate folders
    train_ids, val_ids, test_ids = obtain_ids(src_repo, token, roi_dir)
    ids_by_split = {"train": train_ids, "val": val_ids, "test": test_ids}

    label_sel_by_id = (
        fire_label_sel(download_roi_csvs(src_repo, token, roi_dir))
        if use_roi_label_sel
        else {}
    )
    manifest = read_label_sel_manifest(processor.api, window_repo, token)
    existing_files = set(processor.api.list_repo_files(window_repo, repo_type="dataset"))

    for split_name in splits:
        # Processed fires built with a different label rule are rebuilt.
        stale = []
        for fire_id in ids_by_split[split_name]:
            repo_path = f"{split_name}/p{fire_id}.npz"
            wanted = int(label_sel_by_id.get(str(fire_id), DEFAULT_LABEL_SEL))
            if repo_path in existing_files and manifest.get(repo_path, DEFAULT_LABEL_SEL) != wanted:
                stale.append(str(fire_id))
        if stale:
            print(f"{split_name}: rebuilding {len(stale)} processed fires whose label_sel changed: {stale}")

        written = processor.run_split(
            ids_by_split[split_name],
            split_name,
            length=length,
            interval=interval,
            label_sel_by_id=label_sel_by_id,
            overwrite=overwrite,
            debug=debug,
            reprocess=stale,
        )
        if written:
            manifest.update(written)
            processor.api.upload_file(
                path_or_fileobj=json.dumps(manifest, indent=2, sort_keys=True).encode(),
                path_in_repo=LABEL_SEL_FILE,
                repo_id=window_repo,
                repo_type="dataset",
            )
    return processor


# ----- DATA LOADING ----- #

class DatasetLoader:
    def __init__(self, processed_repo_id, token, train_ids, val_ids, test_ids, base_output_dir="./data", exclude_files=()):
        self.processed_repo_id = processed_repo_id
        self.exclude_files = set(exclude_files)  # repo paths, e.g. 'train/p20778153.npz'
        self.token = token
        self.train_ids = set(train_ids)  # Convert to set for faster lookup
        self.val_ids = set(val_ids)      # Convert to set for faster lookup
        self.test_ids = set(test_ids)    # Convert to set for faster lookup
        self.base_output_dir = base_output_dir
        self.api = HfApi(token=token)

        self.train_dir = os.path.join(self.base_output_dir, "train")
        self.val_dir = os.path.join(self.base_output_dir, "val")
        self.test_dir = os.path.join(self.base_output_dir, "test")

        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.val_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)

    def load_and_organize_data(self):
        print(f"Loading data from: {self.processed_repo_id}")
        all_processed_files = self.api.list_repo_files(self.processed_repo_id, repo_type="dataset")

        for file_path in all_processed_files:
            # Only split folders hold window archives; tessera/p<id>.npz shares
            # their file names and must not be routed into a split folder.
            if file_path.split('/')[0] not in SPLITS:
                continue
            if file_path in self.exclude_files:
                print(f"Skipping {file_path}: excluded")
                if os.path.exists(os.path.join(self.base_output_dir, file_path)):
                    print(f"WARNING: {file_path} is excluded but already exists in {self.base_output_dir}")
                continue
            if file_path.endswith('.npz'):
                # Extract loc_name from filename, e.g., 'train/p20562846.npz' -> '20562846'
                filename_without_extension = os.path.basename(file_path).replace('.npz', '').replace('p', '')
                if filename_without_extension in self.train_ids:
                    target_dir = self.train_dir
                elif filename_without_extension in self.val_ids:
                    target_dir = self.val_dir
                elif filename_without_extension in self.test_ids:
                    target_dir = self.test_dir
                else:
                    print(f"Skipping {file_path}: {filename_without_extension} ID not found in train, val, or test sets.")
                    continue
                local_file_path = os.path.join(target_dir, os.path.basename(file_path))
                if not os.path.exists(local_file_path):
                    print(f"Downloading {file_path} to {target_dir}")
                    downloaded_path = hf_hub_download(
                        repo_id=self.processed_repo_id,
                        filename=file_path,
                        repo_type="dataset",
                        local_dir=self.base_output_dir,
                        token=self.token,
                    )
                    shutil.move(downloaded_path, local_file_path)
                else:
                    print(f"File already exists: {local_file_path}")

        print("Data loading and organization complete.")

    def load_tessera(self, tessera_repo=TESSERA_REPO):
        """Download tessera/* (per-fire embeddings) from `tessera_repo` into <base_output_dir>/tessera."""
        tessera_files = [
            file_path
            for file_path in self.api.list_repo_files(tessera_repo, repo_type="dataset")
            if file_path.startswith("tessera/")
        ]
        print(f"Downloading {len(tessera_files)} TESSERA files")
        for file_path in tessera_files:
            if os.path.exists(os.path.join(self.base_output_dir, file_path)):
                continue
            hf_hub_download(
                repo_id=tessera_repo,
                filename=file_path,
                repo_type="dataset",
                local_dir=self.base_output_dir,
                token=self.token,
            )

        missing = [
            os.path.join(split, file_name)
            for split in SPLITS
            for file_name in sorted(os.listdir(os.path.join(self.base_output_dir, split)))
            if file_name.endswith(".npz")
            and not os.path.exists(os.path.join(self.base_output_dir, "tessera", file_name))
        ]
        if missing:
            print(f"WARNING: {len(missing)} window archives have no TESSERA file: {missing}")


def download_processed_data(
    token,
    window_repo=WINDOW_REPO,
    src_repo=SRC_REPO,
    base_output_dir="./data",
    roi_dir="hf_roi",
    include_tessera=False,
    tessera_repo=TESSERA_REPO,
):
    """Download every windowed NPZ in `window_repo` into <base_output_dir>/{train,val,test}.

    With `include_tessera` (TESSERA mode), fires with no prior-year TESSERA
    (the 2017 fires) are left out and per-fire embeddings are downloaded from
    `tessera_repo` into <base_output_dir>/tessera. The dropped archive names are kept on the
    returned loader as `excluded_npz`. Use a different `base_output_dir` for
    each mode so a full download never mixes with a TESSERA one.
    """
    window_config = read_window_config(HfApi(token=token), window_repo, token)
    print(f"Window repo {window_repo}: {window_config}")
    train_ids, val_ids, test_ids = obtain_ids(src_repo, token, roi_dir)
    excluded_npz = []
    if include_tessera:
        from satimg_dataset_processor.tessera import tessera_excluded_npz

        excluded_npz = tessera_excluded_npz(token, src_repo, roi_dir)
        print(f"TESSERA mode: dropping {len(excluded_npz)} fires without prior-year TESSERA")
    data_loader = DatasetLoader(
        processed_repo_id=window_repo,
        token=token,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        base_output_dir=base_output_dir,
        exclude_files=excluded_npz,
    )
    data_loader.excluded_npz = excluded_npz
    data_loader.load_and_organize_data()
    if include_tessera:
        data_loader.load_tessera(tessera_repo)
    return data_loader


# ----- ZIPPING ----- #

# zip npz files into a zip, creating output_path file
def zip_data_from(folders, output_path):
    zip_path = Path(output_path)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zipf:
        for folder in folders:
            folder_path = Path(folder)

            if not folder_path.exists():
                print(f"Folder {folder_path} does not exist. Skipping.")
                continue

            for file_path in folder_path.rglob('**/*'):
                if file_path.is_file():
                    archive_name = file_path.relative_to(folder_path.parent)
                    zipf.write(file_path, archive_name)
                    print(f"Added {file_path} to zip file as {archive_name}")
                else:
                    print(f"{file_path} not file")


# send data.zip to repo
def zip_data_to(token, zip_path="./data.zip", repo_id=WINDOW_REPO, zip_name="data.zip"):
    return HfApi().upload_file(
        repo_id=repo_id,
        path_in_repo=zip_name,
        path_or_fileobj=str(zip_path),
        repo_type="dataset",
        token=token,
    )


def upload_data_zip(
    token,
    data_root="./data",
    zip_path="./data.zip",
    repo_id=WINDOW_REPO,
    folders=("test", "train", "val", "esri", "tessera"),
    zip_name="data.zip",
):
    """Zip <data_root>/<folder> for each folder and upload it to `repo_id` as `zip_name`.

    TESSERA mode uses repo_id=TESSERA_REPO, zip_name="data_tessera.zip", so
    the full data.zip in WINDOW_REPO is never replaced by the reduced set.
    """
    zip_data_from([os.path.join(data_root, folder) for folder in folders], zip_path)
    return zip_data_to(token, zip_path, repo_id, zip_name)


# pull from processed data the zip file.
def pull_zip_data_to(extraction_path, token=None, repo_id=WINDOW_REPO, download_dir="./data", zip_name="data.zip"):
    zip_file_path = HfApi().hf_hub_download(
        repo_id=repo_id,
        filename=zip_name,
        repo_type="dataset",
        local_dir=download_dir,
        token=token,
    )

    if os.path.exists(zip_file_path):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extraction_path)
        print(f'Successfully extracted {zip_file_path} to {extraction_path}')
    else:
        print(f'Error: {zip_file_path} not found. Please ensure the data is downloaded first.')


def pull_processed_data(data_root, token=None, processed_repo=PROCESSED_REPO):
    """Pull the old-format data.zip from `processed_repo` into `data_root`.

    This is the input to convert_and_upload. Use a different `data_root` than
    the windowed data, since both formats use the same file names.
    """
    pull_zip_data_to(data_root, token=token, repo_id=processed_repo, download_dir=data_root)


# ----- OLD NPZ CONVERSION ----- #

def convert_old_npz(source_path, destination_path):
    """
    Convert:

        data:   (N, C, T, H, W)
        labels: (N, H, W)

    into independently compressed members:

        data_000, label_000
        data_001, label_001
        ...
    """
    with np.load(source_path, allow_pickle=False) as old_archive:
        if "data" not in old_archive or "labels" not in old_archive:
            raise ValueError(
                f"{source_path} does not contain old-format "
                "'data' and 'labels' members"
            )

        data = old_archive["data"]
        labels = old_archive["labels"]

        if len(data) != len(labels):
            raise ValueError(
                f"{source_path}: data contains {len(data)} windows, "
                f"but labels contains {len(labels)}"
            )

        return PredDatasetProcessor._save_windowed_npz(
            destination_path,
            data,
            labels,
        )


def upload_batch(api, staging_dir, split_name, count, window_repo=WINDOW_REPO):
    """Upload one staged batch and remove it after success."""
    if count == 0:
        return

    print(f"Uploading {count} {split_name} archives...")

    api.upload_folder(
        folder_path=str(staging_dir),
        repo_id=window_repo,
        repo_type="dataset",
        path_in_repo=split_name,
    )

    # Upload succeeded, so the temporary converted files can be removed.
    shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    gc.collect()


def convert_and_upload(
    token,
    data_root="./data",
    window_repo=None,
    batch_size=10,
    overwrite_existing=False,
    work_dir=".",
):
    """Convert old-format NPZs under <data_root>/{train,val,test} and upload to `window_repo`.

    The old-format data was built with LEGACY_WINDOW_SETTINGS (10 input days
    every 3 days), so it goes to that settings' window repo by default.
    """
    if window_repo is None:
        window_repo = window_repo_for(*LEGACY_WINDOW_SETTINGS)
    if not token:
        raise RuntimeError("A Hugging Face write token is required")

    data_root = Path(data_root)

    login(token=token)
    api = HfApi(token=token)

    api.create_repo(
        repo_id=window_repo,
        repo_type="dataset",
        exist_ok=True,
    )
    ensure_window_config(api, window_repo, *LEGACY_WINDOW_SETTINGS, token=token)

    existing_remote_files = set(
        api.list_repo_files(
            repo_id=window_repo,
            repo_type="dataset",
        )
    )

    staging_root = Path(
        tempfile.mkdtemp(
            prefix="windowed_npz_",
            dir=work_dir,
        )
    )

    print(f"Temporary staging directory: {staging_root}")

    completed_successfully = False

    try:
        total_converted = 0
        total_skipped = 0
        total_windows = 0

        for split_name in ("train", "val", "test"):
            source_dir = data_root / split_name

            if not source_dir.exists():
                print(f"Skipping missing directory: {source_dir}")
                continue

            source_files = sorted(source_dir.glob("*.npz"))

            if not source_files:
                print(f"No NPZ files found in {source_dir}")
                continue

            split_staging_dir = staging_root / split_name
            split_staging_dir.mkdir(parents=True, exist_ok=True)

            staged_count = 0

            for source_path in tqdm(
                source_files,
                desc=f"Converting {split_name}",
            ):
                remote_path = f"{split_name}/{source_path.name}"

                if (
                    remote_path in existing_remote_files
                    and not overwrite_existing
                ):
                    print(f"Skipping existing remote file: {remote_path}")
                    total_skipped += 1
                    continue

                destination_path = (
                    split_staging_dir / source_path.name
                )

                number_of_windows = convert_old_npz(
                    source_path,
                    destination_path,
                )

                total_converted += 1
                total_windows += number_of_windows
                staged_count += 1

                print(
                    f"Converted {source_path.name}: "
                    f"{number_of_windows} windows"
                )

                # Release decompressed arrays before processing the next file.
                gc.collect()

                if staged_count >= batch_size:
                    upload_batch(
                        api,
                        split_staging_dir,
                        split_name,
                        staged_count,
                        window_repo,
                    )
                    staged_count = 0

            # Upload the final partial batch for this split.
            if staged_count > 0:
                upload_batch(
                    api,
                    split_staging_dir,
                    split_name,
                    staged_count,
                    window_repo,
                )

        completed_successfully = True

        print()
        print("Conversion complete")
        print(f"Archives converted: {total_converted}")
        print(f"Archives skipped:   {total_skipped}")
        print(f"Windows converted:  {total_windows}")
        print(f"Destination:         {window_repo}")

    finally:
        if completed_successfully:
            shutil.rmtree(staging_root, ignore_errors=True)
        else:
            print(
                "Conversion stopped before completion. Temporary files "
                f"were retained at: {staging_root}"
            )
