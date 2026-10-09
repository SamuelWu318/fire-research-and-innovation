"""Build the TS-SatFire next-day prediction dataset through Hugging Face.

Each subcommand wraps one function in satimg_dataset_processor.hf_dataset or
satimg_dataset_processor.tessera. The token defaults to the HF_TOKEN
environment variable. Defaults come from satimg_dataset_processor/config.py:
-ts/-it (window length and stride) and the four dataset repos. --window-repo
defaults to the repo for the chosen -ts/-it (config.window_repo_for).

    python dataset_gen_pred.py process -ts 10 -it 3
    python dataset_gen_pred.py process -ts 6 -it 3      # -> ...-processed-window-ts6-it3
    python dataset_gen_pred.py tessera [--only-fires 21890003 ...]
    python dataset_gen_pred.py tessera-zip
    python dataset_gen_pred.py pull-zip --data-root ./data && python dataset_gen_pred.py pull-tessera --data-root ./data
    python dataset_gen_pred.py download -ts 6 -it 3 --data-root ./data [--tessera]
    python dataset_gen_pred.py upload-zip --data-root ./data [--zip-repo tessera --zip-name data_tessera.zip]
    python dataset_gen_pred.py pull-zip --data-root ./data [--zip-repo processed|window|tessera]
    python dataset_gen_pred.py convert --data-root ./data_processed [--pull]
"""

import argparse
import os

from satimg_dataset_processor.config import TS_INTERVAL, TS_LENGTH, window_repo_for
from satimg_dataset_processor.hf_dataset import (
    PROCESSED_REPO,
    SRC_REPO,
    TESSERA_REPO,
    convert_and_upload,
    download_processed_data,
    process_data,
    pull_processed_data,
    pull_zip_data_to,
    upload_data_zip,
)
from satimg_dataset_processor.tessera import (
    process_tessera,
    pull_tessera_data,
    upload_tessera_zip,
)


ZIP_REPOS = ("window", "processed", "tessera")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token (default: $HF_TOKEN)",
    )
    parser.add_argument("--src-repo", default=SRC_REPO, help="Raw GeoTIFFs + ROI CSVs")
    parser.add_argument("--processed-repo", default=PROCESSED_REPO, help="Old-format NPZ data.zip")
    parser.add_argument(
        "--window-repo",
        "--dest-repo",
        dest="window_repo",
        default=None,
        help="Windowed NPZ per fire (default: the repo for -ts/-it)",
    )
    parser.add_argument("--tessera-repo", default=TESSERA_REPO, help="Prior-year TESSERA per fire")
    parser.add_argument("--roi-dir", default="hf_roi")
    parser.add_argument("--work-dir", default=".")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Window settings, shared by every command that reads or writes windows.
    windows = argparse.ArgumentParser(add_help=False)
    windows.add_argument(
        "-ts",
        "--length",
        dest="ts",
        type=int,
        default=TS_LENGTH,
        help=f"Input days per window (default {TS_LENGTH})",
    )
    windows.add_argument(
        "-it",
        "--interval",
        dest="it",
        type=int,
        default=TS_INTERVAL,
        help=f"Days between window starts (default {TS_INTERVAL})",
    )

    process = subparsers.add_parser(
        "process",
        parents=[windows],
        help="Stream raw fires from --src-repo into windowed NPZs in --window-repo",
    )
    process.add_argument(
        "-mode",
        nargs="+",
        choices=("train", "val", "test"),
        default=["train", "val", "test"],
    )
    process.add_argument("--batch-size", type=int, default=10)
    process.add_argument("--overwrite", action="store_true")
    process.add_argument("--debug", action="store_true")

    tessera = subparsers.add_parser(
        "tessera",
        help="Build prior-year TESSERA embeddings per fire into --tessera-repo",
    )
    tessera.add_argument(
        "-mode",
        nargs="+",
        choices=("train", "val", "test"),
        default=["train", "val", "test"],
    )
    tessera.add_argument("--batch-size", type=int, default=1)
    tessera.add_argument("--strip-rows", type=int, default=128)
    tessera.add_argument("--overwrite", action="store_true")
    tessera.add_argument("--only-fires", nargs="+", help="Restrict to these fire IDs (timing test)")

    subparsers.add_parser(
        "tessera-zip",
        help="Bundle --tessera-repo tessera/ and the drop list into tessera.zip there",
    )
    pull_tessera = subparsers.add_parser(
        "pull-tessera",
        help="Unpack tessera.zip into --data-root/tessera and drop 2017 window archives",
    )
    pull_tessera.add_argument("--data-root", default="./data")
    pull_tessera.add_argument("--keep-excluded", action="store_true")

    download = subparsers.add_parser(
        "download",
        parents=[windows],
        help="Download --window-repo NPZs into --data-root/{train,val,test}",
    )
    download.add_argument("--data-root", default="./data")
    download.add_argument(
        "--tessera",
        action="store_true",
        help="TESSERA mode: drop fires without prior-year TESSERA and download "
        "--tessera-repo embeddings into --data-root/tessera",
    )

    upload_zip = subparsers.add_parser(
        "upload-zip",
        parents=[windows],
        help="Zip --data-root and upload it to --zip-repo",
    )
    upload_zip.add_argument("--data-root", default="./data")
    upload_zip.add_argument("--zip-path", default="./data.zip")
    upload_zip.add_argument("--zip-repo", choices=ZIP_REPOS, default="window")
    upload_zip.add_argument("--zip-name", default="data.zip", help="data_tessera.zip for TESSERA mode")

    pull_zip = subparsers.add_parser(
        "pull-zip",
        parents=[windows],
        help="Download a zip from --zip-repo and extract it into --data-root",
    )
    pull_zip.add_argument("--data-root", default="./data")
    pull_zip.add_argument("--zip-repo", choices=ZIP_REPOS, default="window")
    pull_zip.add_argument("--zip-name", default="data.zip")

    convert = subparsers.add_parser(
        "convert",
        help="Convert old-format NPZs under --data-root and upload them to --window-repo",
    )
    convert.add_argument("--data-root", default="./data_processed")
    convert.add_argument(
        "--pull",
        action="store_true",
        help="First pull data.zip from --processed-repo into --data-root",
    )
    convert.add_argument("--batch-size", type=int, default=10)
    convert.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.window_repo is None and args.command in ("process", "download", "upload-zip", "pull-zip"):
        args.window_repo = window_repo_for(args.ts, args.it)
    zip_repos = {
        "window": args.window_repo,
        "processed": args.processed_repo,
        "tessera": args.tessera_repo,
    }

    if args.command == "process":
        process_data(
            args.token,
            src_repo=args.src_repo,
            window_repo=args.window_repo,
            batch_size=args.batch_size,
            length=args.ts,
            interval=args.it,
            splits=args.mode,
            roi_dir=args.roi_dir,
            work_dir=args.work_dir,
            overwrite=args.overwrite,
            debug=args.debug,
        )
    elif args.command == "tessera":
        process_tessera(
            args.token,
            src_repo=args.src_repo,
            tessera_repo=args.tessera_repo,
            splits=args.mode,
            batch_size=args.batch_size,
            work_dir=args.work_dir,
            roi_dir=args.roi_dir,
            overwrite=args.overwrite,
            strip_rows=args.strip_rows,
            only_fires=args.only_fires,
        )
    elif args.command == "tessera-zip":
        upload_tessera_zip(
            args.token,
            src_repo=args.src_repo,
            tessera_repo=args.tessera_repo,
            work_dir=args.work_dir,
            roi_dir=args.roi_dir,
        )
    elif args.command == "pull-tessera":
        pull_tessera_data(
            args.data_root,
            token=args.token,
            tessera_repo=args.tessera_repo,
            drop_excluded=not args.keep_excluded,
        )
    elif args.command == "download":
        download_processed_data(
            args.token,
            window_repo=args.window_repo,
            src_repo=args.src_repo,
            base_output_dir=args.data_root,
            roi_dir=args.roi_dir,
            include_tessera=args.tessera,
            tessera_repo=args.tessera_repo,
        )
    elif args.command == "upload-zip":
        upload_data_zip(
            args.token,
            data_root=args.data_root,
            zip_path=args.zip_path,
            repo_id=zip_repos[args.zip_repo],
            zip_name=args.zip_name,
        )
    elif args.command == "pull-zip":
        # The zips store their folders (train/, val/, test/, ...) at the root.
        pull_zip_data_to(
            args.data_root,
            token=args.token,
            repo_id=zip_repos[args.zip_repo],
            download_dir=args.data_root,
            zip_name=args.zip_name,
        )
    elif args.command == "convert":
        if args.pull:
            pull_processed_data(args.data_root, token=args.token, processed_repo=args.processed_repo)
        convert_and_upload(
            args.token,
            data_root=args.data_root,
            window_repo=args.window_repo,
            batch_size=args.batch_size,
            overwrite_existing=args.overwrite,
            work_dir=args.work_dir,
        )
