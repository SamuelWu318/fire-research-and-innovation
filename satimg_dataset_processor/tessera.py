"""Prior-year TESSERA embeddings aligned to each fire's VIIRS crop.

PredDatasetProcessor.process_location crops rows/cols 128:384 of every
VIIRS_Day GeoTIFF, so one 256x256 grid covers all of a fire's windows. For
each fire this module reads that grid's georeference from the raw GeoTIFFs in
the source Hugging Face repo, streams the TESSERA embeddings covering it, and
averages every 10 m TESSERA pixel into the crop pixel containing its centre.
The result is one tessera/p<id>.npz per fire in the TESSERA repo
(SamuelWu318/ts-tesserafire), separate from the processed window repo.
upload_tessera_zip then bundles them, with the TESSERA-mode drop list, into
tessera.zip in the same repo; pull_tessera_data unpacks it next to data.zip.
Each file holds:

    embedding      (128, 256, 256) float16, 0 where coverage is 0
    coverage       (256, 256) float16, fraction of 10 m pixels with data
    fire_year, tessera_year, crs, transform, bounds, bounds_lonlat,
    source_store, model_version, geotessera_version, code_version, day_files

Embeddings come from the year before the fire: an annual embedding for the
fire year is built partly from post-fire imagery and would leak the burn scar
into the inputs. Fires whose prior year is not published (2017 fires) get no
file.

Everything is fetched into `work_dir` or streamed into RAM; nothing is cached
outside the paths passed in.
"""

import gc
import os
import shutil
import subprocess
import time
import traceback
from importlib.metadata import version as package_version

import numpy as np
import pandas as pd
import rasterio
from huggingface_hub import HfApi, hf_hub_download, login
from pyproj import Transformer
from rasterio.transform import Affine, array_bounds
from rasterio.warp import transform_bounds

from satimg_dataset_processor.hf_dataset import (
    SRC_REPO,
    TESSERA_REPO,
    download_roi_csvs,
    list_fire_files,
    obtain_ids,
    pull_zip_data_to,
    zip_data_from,
    zip_data_to,
)


FORMAT_VERSION = 1
N_BANDS = 128
# Must match the crop in PredDatasetProcessor.process_location.
CROP_OFFSET = 128
CROP_SIZE = 256
# First year in the v1.1 dClimate store (GeoTesseraZarr().years starts here).
FIRST_TESSERA_YEAR = 2017
# Drop list shipped inside tessera.zip, one window archive path per line.
EXCLUDED_LIST = "excluded_npz.txt"


def has_prior_year_tessera(fire_year):
    """True if the year before `fire_year` has published TESSERA embeddings."""
    return fire_year is not None and int(fire_year) - 1 >= FIRST_TESSERA_YEAR


def code_version():
    """Git commit of this repository, or "" when not run from a checkout."""
    try:
        return subprocess.run(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return ""


def crop_grid(profile, offset=CROP_OFFSET, size=CROP_SIZE):
    """Return (crs, transform) of the crop process_location takes from a GeoTIFF."""
    if profile["height"] < offset + size or profile["width"] < offset + size:
        raise ValueError(
            f"GeoTIFF is {profile['height']}x{profile['width']}; the "
            f"{size}x{size} crop at offset {offset} does not fit"
        )
    return profile["crs"], profile["transform"] * Affine.translation(offset, offset)


def utm_zones(west, east):
    """UTM zone numbers spanned by a longitude range (no antimeridian wrap)."""
    first = int(np.floor((west + 180) / 6)) + 1
    last = int(np.floor((np.nextafter(east, west) + 180) / 6)) + 1
    return list(range(max(1, first), min(60, last) + 1))


def grid_tessera(gt, dst_crs, dst_transform, year, size=CROP_SIZE, strip_rows=128):
    """Area-average one year of TESSERA onto a size x size raster grid.

    `gt` is a geotessera.GeoTesseraZarr. The region is streamed in row strips
    so a full-resolution crop (about 9600 x 9600 x 128 values) never sits in
    memory at once. Each 10 m pixel is assigned to the grid cell containing
    its centre; pixels without data (water, gaps) are left out of the mean.

    Returns (embedding, coverage, source_pixels, bounds_lonlat):
        embedding      (128, size, size) float32, NaN where coverage is 0
        coverage       (size, size) float32, valid / total 10 m pixels
        source_pixels  (size, size) int64, total 10 m pixels per cell
    """
    n_cells = size * size
    sums = np.zeros((N_BANDS, n_cells), dtype=np.float64)
    valid_counts = np.zeros(n_cells, dtype=np.int64)
    total_counts = np.zeros(n_cells, dtype=np.int64)

    bounds_lonlat = transform_bounds(
        dst_crs,
        "EPSG:4326",
        *array_bounds(size, size, dst_transform),
        densify_pts=21,
    )
    west, south, east, north = bounds_lonlat
    inverse = ~dst_transform
    zones = utm_zones(west, east)

    for zone in zones:
        # Each zone serves only its own longitudes, so a crop on a zone seam
        # is read once per side without counting any pixel twice.
        zone_west, zone_east = zone * 6 - 186, zone * 6 - 180
        bbox = (max(west, zone_west), south, min(east, zone_east), north)
        to_dst = to_lonlat = None

        for block, transform, zone_crs in gt.iter_region(bbox, year, strip_rows=strip_rows):
            if to_dst is None:
                to_dst = Transformer.from_crs(zone_crs, dst_crs, always_xy=True)
                to_lonlat = Transformer.from_crs(zone_crs, "EPSG:4326", always_xy=True)

            rows, cols = block.shape[:2]
            col_centres, row_centres = np.meshgrid(
                np.arange(cols) + 0.5,
                np.arange(rows) + 0.5,
            )
            xs = transform.c + transform.a * col_centres + transform.b * row_centres
            ys = transform.f + transform.d * col_centres + transform.e * row_centres

            px, py = to_dst.transform(xs, ys)
            dst_col = np.floor(inverse.a * px + inverse.b * py + inverse.c)
            dst_row = np.floor(inverse.d * px + inverse.e * py + inverse.f)
            inside = (
                (dst_col >= 0) & (dst_col < size)
                & (dst_row >= 0) & (dst_row < size)
            )
            if len(zones) > 1:
                lon, _ = to_lonlat.transform(xs, ys)
                inside &= (lon >= zone_west) & (lon < zone_east)
            if not inside.any():
                continue

            cell = (dst_row[inside] * size + dst_col[inside]).astype(np.int64)
            values = block[inside]
            has_data = np.isfinite(values).all(axis=1)
            total_counts += np.bincount(cell, minlength=n_cells)

            cell = cell[has_data]
            values = values[has_data]
            valid_counts += np.bincount(cell, minlength=n_cells)
            for band in range(N_BANDS):
                sums[band] += np.bincount(
                    cell,
                    weights=values[:, band],
                    minlength=n_cells,
                )

            del block, values, xs, ys, px, py
            gc.collect()

    embedding = np.full((N_BANDS, n_cells), np.nan, dtype=np.float32)
    has_cell_data = valid_counts > 0
    embedding[:, has_cell_data] = sums[:, has_cell_data] / valid_counts[has_cell_data]
    coverage = np.divide(
        valid_counts,
        total_counts,
        out=np.zeros(n_cells),
        where=total_counts > 0,
    )

    return (
        embedding.reshape(N_BANDS, size, size),
        coverage.reshape(size, size).astype(np.float32),
        total_counts.reshape(size, size),
        bounds_lonlat,
    )


def save_tessera_npz(save_path, embedding, coverage, metadata):
    """Write one fire's embedding, coverage, and metadata, then verify it."""
    if embedding.shape != (N_BANDS, CROP_SIZE, CROP_SIZE):
        raise ValueError(f"Unexpected embedding shape {embedding.shape}")
    if coverage.shape != (CROP_SIZE, CROP_SIZE):
        raise ValueError(f"Unexpected coverage shape {coverage.shape}")

    stored = np.nan_to_num(embedding, nan=0.0).astype(np.float16)
    if not np.isfinite(stored).all():
        raise ValueError(f"{save_path}: embedding overflows float16")

    np.savez_compressed(
        save_path,
        format_version=np.asarray(FORMAT_VERSION, dtype=np.int16),
        embedding=stored,
        coverage=coverage.astype(np.float16),
        **{key: np.asarray(value) for key, value in metadata.items()},
    )

    with np.load(save_path, allow_pickle=False) as archive:
        if archive["embedding"].shape != (N_BANDS, CROP_SIZE, CROP_SIZE):
            raise RuntimeError(f"Failed to verify {save_path}")


def build_fire_tessera(gt, fire_id, day_paths, fire_year, save_path, strip_rows=128):
    """Build tessera/p<fire_id>.npz from local VIIRS_Day GeoTIFFs.

    `day_paths` maps source repo names to local paths. Every given file must
    share one georeference, otherwise the fixed crop would cover different
    ground on different days and no single embedding could match it.
    """
    profiles = {}
    for remote_name, local_path in day_paths.items():
        with rasterio.open(local_path) as reader:
            profiles[remote_name] = reader.profile

    reference_name, reference = next(iter(profiles.items()))
    for remote_name, profile in profiles.items():
        if profile["crs"] != reference["crs"] or not profile["transform"].almost_equals(
            reference["transform"]
        ):
            raise ValueError(
                f"{fire_id}: georeference of {remote_name} differs from "
                f"{reference_name}"
            )

    dst_crs, dst_transform = crop_grid(reference)
    tessera_year = int(fire_year) - 1
    embedding, coverage, _, bounds_lonlat = grid_tessera(
        gt,
        dst_crs,
        dst_transform,
        tessera_year,
        strip_rows=strip_rows,
    )

    metadata = {
        "fire_id": str(fire_id),
        "fire_year": np.int16(fire_year),
        "tessera_year": np.int16(tessera_year),
        "crs": dst_crs.to_wkt(),
        "transform": np.asarray(tuple(dst_transform)[:6], dtype=np.float64),
        "bounds": np.asarray(array_bounds(CROP_SIZE, CROP_SIZE, dst_transform), dtype=np.float64),
        "bounds_lonlat": np.asarray(bounds_lonlat, dtype=np.float64),
        "source_store": str(gt.url),
        "model_version": str(getattr(gt, "model_version", "")),
        "geotessera_version": package_version("geotessera"),
        "code_version": code_version(),
        "day_files": np.asarray(sorted(day_paths), dtype=str),
    }
    save_tessera_npz(save_path, embedding, coverage, metadata)

    print(
        f"{fire_id}: TESSERA {tessera_year}, mean coverage "
        f"{coverage.mean():.3f}, cells without data "
        f"{int(np.count_nonzero(coverage == 0))}/{coverage.size}"
    )
    return coverage


def fire_start_years(roi_paths):
    """Map fire Id -> start year from the ROI CSVs."""
    years = {}
    for path in roi_paths:
        df = pd.read_csv(path, dtype={"Id": str})
        if "Id" not in df.columns or "start_date" not in df.columns:
            continue
        for fire_id, start_date in zip(df["Id"], df["start_date"]):
            if pd.notna(start_date):
                years[str(fire_id)] = int(pd.to_datetime(start_date).year)
    return years


def tessera_excluded_npz(token=None, src_repo=SRC_REPO, roi_dir="hf_roi"):
    """Window archives to drop in TESSERA mode, e.g. ['train/p20778153.npz', ...].

    These are the fires with no prior-year TESSERA (all 2017 fires). Dropping
    them from both the TESSERA and the matched no-TESSERA runs keeps the
    comparison on the same fires.
    """
    years = fire_start_years(download_roi_csvs(src_repo, token, roi_dir))
    ids_by_split = dict(zip(("train", "val", "test"), obtain_ids(src_repo, token, roi_dir)))
    return sorted(
        f"{split}/p{fire_id}.npz"
        for split, fire_ids in ids_by_split.items()
        for fire_id in fire_ids
        if not has_prior_year_tessera(years.get(str(fire_id)))
    )


def process_tessera(
    token,
    src_repo=SRC_REPO,
    tessera_repo=TESSERA_REPO,
    splits=("train", "val", "test"),
    batch_size=1,
    work_dir=".",
    roi_dir="hf_roi",
    overwrite=False,
    store_url=None,
    strip_rows=128,
    only_fires=None,
):
    """Build tessera/p<id>.npz for every fire in `splits` and upload them to `tessera_repo`.

    `only_fires` restricts the run to those fire IDs, e.g. a short timing test.

    Only the first and last VIIRS_Day GeoTIFF of each fire are downloaded
    (into <work_dir>/temp, deleted after each fire); embeddings are streamed
    into RAM. Finished files are staged in <work_dir>/staging/tessera and
    uploaded to `tessera_repo` in batches, so an interrupted run resumes by
    skipping fires already uploaded. Returns {fire_id: reason} for skipped
    fires.
    """
    from geotessera import GeoTesseraZarr

    login(token)
    api = HfApi(token=token)
    # A new repo is created private; an existing one keeps its visibility.
    api.create_repo(repo_id=tessera_repo, repo_type="dataset", private=True, exist_ok=True)

    ids_by_split = dict(zip(("train", "val", "test"), obtain_ids(src_repo, token, roi_dir)))
    years = fire_start_years(download_roi_csvs(src_repo, token, roi_dir))
    locations = list_fire_files(api, src_repo)
    existing_files = set(api.list_repo_files(tessera_repo, repo_type="dataset"))

    # No cache_dir: the store is read over HTTP straight into memory.
    gt = GeoTesseraZarr() if store_url is None else GeoTesseraZarr(store_url)
    print(f"TESSERA store {gt.url}, years {gt.years}")

    temp_dir = os.path.join(work_dir, "temp")
    staging_dir = os.path.join(work_dir, "staging", "tessera")
    os.makedirs(staging_dir, exist_ok=True)

    fire_ids = list(dict.fromkeys(
        str(fire_id) for split in splits for fire_id in ids_by_split[split]
    ))
    if only_fires is not None:
        wanted = {str(fire_id) for fire_id in only_fires}
        unknown = wanted.difference(fire_ids)
        if unknown:
            raise ValueError(f"Fires not in splits {splits}: {sorted(unknown)}")
        fire_ids = [fire_id for fire_id in fire_ids if fire_id in wanted]
    skipped = {}
    staged_count = 0

    def upload_staged():
        print(f"Uploading batch of {staged_count} TESSERA files to Hugging Face")
        api.upload_folder(
            folder_path=staging_dir,
            repo_id=tessera_repo,
            repo_type="dataset",
            path_in_repo="tessera",
        )
        shutil.rmtree(staging_dir, ignore_errors=True)
        os.makedirs(staging_dir, exist_ok=True)

    for fire_idx, fire_id in enumerate(fire_ids):
        if f"tessera/p{fire_id}.npz" in existing_files and not overwrite:
            print(f"Skipping {fire_id}: already processed")
            continue

        day_files = locations.get(fire_id, {}).get("VIIRS_Day", [])
        if not day_files:
            skipped[fire_id] = "no VIIRS_Day files in source repo"
            continue
        if fire_id not in years:
            skipped[fire_id] = "no start_date in ROI CSVs"
            continue
        if not has_prior_year_tessera(years[fire_id]) or years[fire_id] - 1 not in gt.years:
            skipped[fire_id] = f"no TESSERA for prior year {years[fire_id] - 1}"
            continue

        print(f"[TESSERA | {fire_idx + 1}/{len(fire_ids)}] Processing: {fire_id}")
        started = time.time()
        try:
            os.makedirs(temp_dir, exist_ok=True)
            day_paths = {
                remote_name: hf_hub_download(
                    repo_id=src_repo,
                    filename=remote_name,
                    repo_type="dataset",
                    cache_dir=temp_dir,
                    token=token,
                )
                for remote_name in dict.fromkeys((day_files[0], day_files[-1]))
            }
            build_fire_tessera(
                gt,
                fire_id,
                day_paths,
                years[fire_id],
                os.path.join(staging_dir, f"p{fire_id}.npz"),
                strip_rows=strip_rows,
            )
            staged_count += 1
            print(f"{fire_id}: built in {(time.time() - started) / 60:.1f} min")
        except Exception as error:
            traceback.print_exc()
            skipped[fire_id] = f"{type(error).__name__}: {error}"
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            gc.collect()

        if staged_count >= batch_size:
            upload_staged()
            staged_count = 0

    if staged_count > 0:
        upload_staged()

    print(f"TESSERA done; skipped {len(skipped)} fires")
    for fire_id, reason in skipped.items():
        print(f"  {fire_id}: {reason}")
    return skipped


def upload_tessera_zip(
    token,
    src_repo=SRC_REPO,
    tessera_repo=TESSERA_REPO,
    work_dir=".",
    roi_dir="hf_roi",
    zip_name="tessera.zip",
):
    """Bundle tessera/*.npz from `tessera_repo` into `zip_name` and upload it there.

    The zip holds tessera/p<id>.npz plus tessera/excluded_npz.txt, the window
    archives of fires without prior-year TESSERA. Window data is not copied:
    training pulls data.zip from WINDOW_REPO and this zip into the same folder.
    """
    bundle_root = os.path.join(work_dir, "tessera_bundle")
    tessera_dir = os.path.join(bundle_root, "tessera")
    os.makedirs(tessera_dir, exist_ok=True)

    api = HfApi(token=token)
    remote_files = [
        file_path
        for file_path in api.list_repo_files(tessera_repo, repo_type="dataset")
        if file_path.startswith("tessera/") and file_path.endswith(".npz")
    ]
    if not remote_files:
        raise RuntimeError(f"No tessera/*.npz in {tessera_repo}; run process_tessera first")

    print(f"Downloading {len(remote_files)} TESSERA files from {tessera_repo}")
    for file_path in remote_files:
        if not os.path.exists(os.path.join(bundle_root, file_path)):
            hf_hub_download(
                repo_id=tessera_repo,
                filename=file_path,
                repo_type="dataset",
                local_dir=bundle_root,
                token=token,
            )

    excluded = tessera_excluded_npz(token, src_repo, roi_dir)
    with open(os.path.join(tessera_dir, EXCLUDED_LIST), "w") as handle:
        handle.write("\n".join(excluded) + "\n")

    # Every fire that is not dropped needs an embedding before training.
    ids_by_split = dict(zip(("train", "val", "test"), obtain_ids(src_repo, token, roi_dir)))
    present = {os.path.basename(file_path) for file_path in remote_files}
    missing = sorted(
        f"{split}/p{fire_id}.npz"
        for split, fire_ids in ids_by_split.items()
        for fire_id in fire_ids
        if f"{split}/p{fire_id}.npz" not in excluded and f"p{fire_id}.npz" not in present
    )
    if missing:
        print(f"WARNING: {len(missing)} kept fires have no TESSERA file yet: {missing}")

    zip_path = os.path.join(work_dir, zip_name)
    zip_data_from([tessera_dir], zip_path)
    zip_data_to(token, zip_path, tessera_repo, zip_name)
    print(
        f"Uploaded {zip_name} to {tessera_repo}: {len(remote_files)} embeddings, "
        f"{len(excluded)} excluded window archives"
    )
    return missing


def pull_tessera_data(
    data_root,
    token=None,
    tessera_repo=TESSERA_REPO,
    zip_name="tessera.zip",
    drop_excluded=True,
):
    """Unpack tessera.zip into <data_root>/tessera and drop excluded windows.

    Run after pulling data.zip into the same `data_root`. With `drop_excluded`,
    the window archives listed in tessera/excluded_npz.txt (fires without
    prior-year TESSERA) are deleted, so both the TESSERA run and the matched
    no-TESSERA run see the same fires.
    """
    pull_zip_data_to(
        data_root,
        token=token,
        repo_id=tessera_repo,
        download_dir=data_root,
        zip_name=zip_name,
    )
    if not drop_excluded:
        return []

    with open(os.path.join(data_root, "tessera", EXCLUDED_LIST)) as handle:
        excluded = [line.strip() for line in handle if line.strip()]
    removed = []
    for name in excluded:
        path = os.path.join(data_root, name)
        if os.path.exists(path):
            os.remove(path)
            removed.append(name)
    print(
        f"TESSERA mode: removed {len(removed)} window archives of fires "
        "without prior-year TESSERA"
    )
    return removed
