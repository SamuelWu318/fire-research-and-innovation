"""Build the TS-SatFire next-day prediction dataset through Hugging Face.

Each subcommand wraps one function in satimg_dataset_processor.hf_dataset.
The token defaults to the HF_TOKEN environment variable.

    python dataset_gen_pred.py process -ts 10 -it 3
    python dataset_gen_pred.py download --data-root ./data
    python dataset_gen_pred.py upload-zip --data-root ./data
    python dataset_gen_pred.py pull-zip --data-root ./data
    python dataset_gen_pred.py convert --data-root ./data
"""

import argparse
import os

from satimg_dataset_processor.hf_dataset import (
    DEST_REPO,
    SRC_REPO,
    convert_and_upload,
    download_processed_data,
    process_data,
    pull_zip_data_to,
    upload_data_zip,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token (default: $HF_TOKEN)",
    )
    parser.add_argument("--src-repo", default=SRC_REPO)
    parser.add_argument("--dest-repo", default=DEST_REPO)
    parser.add_argument("--roi-dir", default="hf_roi")
    parser.add_argument("--work-dir", default=".")
    subparsers = parser.add_subparsers(dest="command", required=True)

    process = subparsers.add_parser(
        "process",
        help="Stream raw fires into windowed NPZs in --dest-repo",
    )
    process.add_argument("-ts", type=int, default=10, help="Length of TS")
    process.add_argument("-it", type=int, default=3, help="Interval")
    process.add_argument(
        "-mode",
        nargs="+",
        choices=("train", "val", "test"),
        default=["train", "val", "test"],
    )
    process.add_argument("--batch-size", type=int, default=10)
    process.add_argument("--overwrite", action="store_true")
    process.add_argument("--debug", action="store_true")

    download = subparsers.add_parser(
        "download",
        help="Download processed NPZs into --data-root/{train,val,test}",
    )
    download.add_argument("--data-root", default="./data")

    upload_zip = subparsers.add_parser(
        "upload-zip",
        help="Zip --data-root and upload it as data.zip",
    )
    upload_zip.add_argument("--data-root", default="./data")
    upload_zip.add_argument("--zip-path", default="./data.zip")

    pull_zip = subparsers.add_parser(
        "pull-zip",
        help="Download data.zip and extract it into --data-root",
    )
    pull_zip.add_argument("--data-root", default="./data")

    convert = subparsers.add_parser(
        "convert",
        help="Convert old-format NPZs under --data-root and upload them",
    )
    convert.add_argument("--data-root", default="./data")
    convert.add_argument("--batch-size", type=int, default=10)
    convert.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.command == "process":
        process_data(
            args.token,
            src_repo=args.src_repo,
            dest_repo=args.dest_repo,
            batch_size=args.batch_size,
            length=args.ts,
            interval=args.it,
            splits=args.mode,
            roi_dir=args.roi_dir,
            work_dir=args.work_dir,
            overwrite=args.overwrite,
            debug=args.debug,
        )
    elif args.command == "download":
        download_processed_data(
            args.token,
            dest_repo=args.dest_repo,
            src_repo=args.src_repo,
            base_output_dir=args.data_root,
            roi_dir=args.roi_dir,
        )
    elif args.command == "upload-zip":
        upload_data_zip(
            args.token,
            data_root=args.data_root,
            zip_path=args.zip_path,
            dest_repo=args.dest_repo,
        )
    elif args.command == "pull-zip":
        # data.zip stores train/, val/, test/ at its root.
        pull_zip_data_to(
            args.data_root,
            token=args.token,
            dest_repo=args.dest_repo,
            download_dir=args.data_root,
        )
    elif args.command == "convert":
        convert_and_upload(
            args.token,
            data_root=args.data_root,
            dest_repo=args.dest_repo,
            batch_size=args.batch_size,
            overwrite_existing=args.overwrite,
            work_dir=args.work_dir,
        )
