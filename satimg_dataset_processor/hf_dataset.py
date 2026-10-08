"""Hugging Face pipeline for the TS-SatFire next-day prediction dataset.

Ported from the SwinUNETR_data_grab Colab notebook. Repositories, tokens, and
working directories are arguments rather than notebook globals, so a Colab
notebook only has to read its secrets and call these functions.

Typical flow:
    process_data(...)            raw GeoTIFF repo -> windowed NPZ repo
    download_processed_data(...) windowed NPZ repo -> <data>/{train,val,test}
    upload_data_zip(...)         <data> -> data.zip in the NPZ repo
    pull_zip_data_to(...)        data.zip -> <data>/{train,val,test}
    convert_and_upload(...)      old-format NPZ -> windowed NPZ repo
"""

import gc
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download, login
from tqdm.auto import tqdm

from satimg_dataset_processor.satimg_dataset_processor import PredDatasetProcessor


SRC_REPO = "SamuelWu318/ts-satfire"
DEST_REPO = "SamuelWu318/ts-satfire-processed-window"

VAL_IDS = [
    '20568194', '20701026', '20562846', '20700973', '24462610',
    '24462788', '24462753', '24103571', '21998313', '21751303',
    '22141596', '21999381', '23301962', '22712904', '22713339',
]
TRAIN_VAL_YEARS = ('2017', '2018', '2019', '2020')
TEST_YEARS = ('2021',)


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
        all_files = self.api.list_repo_files(
            self.src_repo,
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

    def run_split(
        self,
        target_ids,
        split_name,
        length=10,
        interval=3,
        label_sel_by_id=None,
        overwrite=False,
        debug=False,
    ):
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

            if dest_filename in existing_files and not overwrite:
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

            label_sel = 1
            if label_sel_by_id is not None:
                label_sel = int(label_sel_by_id.get(loc_name, 1))

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
                )
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


# ----- SPLITTING ----- #

def obtain_ids(src_repo=SRC_REPO, token=None, roi_dir="hf_roi"):
    """Download the ROI CSVs from `src_repo` and split fire IDs.

    Train/val come from 2017-2020 fires (val is the fixed VAL_IDS list);
    test comes from 2021 fires.
    """
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


def process_data(
    token,
    src_repo=SRC_REPO,
    dest_repo=DEST_REPO,
    batch_size=10,
    length=10,
    interval=3,
    splits=("train", "val", "test"),
    roi_dir="hf_roi",
    work_dir=".",
    overwrite=False,
    debug=False,
):
    """Stream raw fires from `src_repo` into windowed NPZs in `dest_repo`.

    `token` needs write access to `dest_repo`. Fires already present in
    `dest_repo` are skipped unless `overwrite` is set.
    """
    processor = DatasetStreamer(
        src_repo=src_repo,
        dest_repo=dest_repo,
        token=token,
        batch_size=batch_size,
        work_dir=work_dir,
    )

    # using id metadata from above, split into appropriate folders
    train_ids, val_ids, test_ids = obtain_ids(src_repo, token, roi_dir)
    ids_by_split = {"train": train_ids, "val": val_ids, "test": test_ids}
    for split_name in splits:
        processor.run_split(
            ids_by_split[split_name],
            split_name,
            length=length,
            interval=interval,
            overwrite=overwrite,
            debug=debug,
        )
    return processor


# ----- DATA LOADING ----- #

class DatasetLoader:
    def __init__(self, processed_repo_id, token, train_ids, val_ids, test_ids, base_output_dir="./data"):
        self.processed_repo_id = processed_repo_id
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


def download_processed_data(
    token,
    dest_repo=DEST_REPO,
    src_repo=SRC_REPO,
    base_output_dir="./data",
    roi_dir="hf_roi",
):
    """Download every processed NPZ into <base_output_dir>/{train,val,test}."""
    train_ids, val_ids, test_ids = obtain_ids(src_repo, token, roi_dir)
    data_loader = DatasetLoader(
        processed_repo_id=dest_repo,
        token=token,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        base_output_dir=base_output_dir,
    )
    data_loader.load_and_organize_data()
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
def zip_data_to(token, zip_path="./data.zip", dest_repo=DEST_REPO):
    return HfApi().upload_file(
        repo_id=dest_repo,
        path_in_repo="data.zip",
        path_or_fileobj=str(zip_path),
        repo_type="dataset",
        token=token,
    )


def upload_data_zip(
    token,
    data_root="./data",
    zip_path="./data.zip",
    dest_repo=DEST_REPO,
    folders=("test", "train", "val", "esri"),
):
    """Zip <data_root>/<folder> for each folder and upload it as data.zip."""
    zip_data_from([os.path.join(data_root, folder) for folder in folders], zip_path)
    return zip_data_to(token, zip_path, dest_repo)


# pull from processed data the zip file.
def pull_zip_data_to(extraction_path, token=None, dest_repo=DEST_REPO, download_dir="./data"):
    zip_file_path = HfApi().hf_hub_download(
        repo_id=dest_repo,
        filename="data.zip",
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


def upload_batch(api, staging_dir, split_name, count, dest_repo=DEST_REPO):
    """Upload one staged batch and remove it after success."""
    if count == 0:
        return

    print(f"Uploading {count} {split_name} archives...")

    api.upload_folder(
        folder_path=str(staging_dir),
        repo_id=dest_repo,
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
    dest_repo=DEST_REPO,
    batch_size=10,
    overwrite_existing=False,
    work_dir=".",
):
    """Convert old-format NPZs under <data_root>/{train,val,test} and upload."""
    if not token:
        raise RuntimeError("A Hugging Face write token is required")

    data_root = Path(data_root)

    login(token=token)
    api = HfApi(token=token)

    api.create_repo(
        repo_id=dest_repo,
        repo_type="dataset",
        exist_ok=True,
    )

    existing_remote_files = set(
        api.list_repo_files(
            repo_id=dest_repo,
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
                        dest_repo,
                    )
                    staged_count = 0

            # Upload the final partial batch for this split.
            if staged_count > 0:
                upload_batch(
                    api,
                    split_staging_dir,
                    split_name,
                    staged_count,
                    dest_repo,
                )

        completed_successfully = True

        print()
        print("Conversion complete")
        print(f"Archives converted: {total_converted}")
        print(f"Archives skipped:   {total_skipped}")
        print(f"Windows converted:  {total_windows}")
        print(f"Destination:         {dest_repo}")

    finally:
        if completed_successfully:
            shutil.rmtree(staging_root, ignore_errors=True)
        else:
            print(
                "Conversion stopped before completion. Temporary files "
                f"were retained at: {staging_root}"
            )
