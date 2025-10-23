"""
Helper script to download the Kaggle fruit ripeness dataset into the repository.

Usage:
    python Dataset/download_dataset.py

Requirements:
    - kagglehub (pip install kagglehub)
    - A valid Kaggle API token at ~/.kaggle/kaggle.json or in the current directory
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import kagglehub

DEFAULT_DATASET = "leftin/fruit-ripeness-unripe-ripe-and-rotten"
DEFAULT_OUTPUT_DIR = Path("Dataset") / "fruit_ripeness_dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the fruit ripeness dataset from Kaggle using kagglehub.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Kaggle dataset slug in the form owner/dataset-name",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the dataset should be copied/extracted",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the target directory if it already contains files",
    )
    parser.add_argument(
        "--keep-zips",
        action="store_true",
        help="Keep the original .zip archives after extraction",
    )
    return parser.parse_args()


def ensure_empty_dir(path: Path, force: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not force:
            raise RuntimeError(
                f"Output directory '{path}' is not empty. Use --force to overwrite."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_dataset(src: Path, dst: Path) -> None:
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def extract_archives(path: Path, keep_zips: bool) -> None:
    zip_files = list(path.rglob("*.zip"))
    for zip_path in zip_files:
        extract_dir = zip_path.with_suffix("")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        if not keep_zips:
            zip_path.unlink()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()

    print(f"Downloading Kaggle dataset '{args.dataset}' with kagglehub...")
    downloaded_path = Path(kagglehub.dataset_download(args.dataset)).resolve()
    if not downloaded_path.exists():
        raise FileNotFoundError(
            f"kagglehub reported download path '{downloaded_path}', but it does not exist."
        )
    print(f"Download complete. Source content located at: {downloaded_path}")

    ensure_empty_dir(output_dir, args.force)
    print(f"Copying dataset into: {output_dir}")
    copy_dataset(downloaded_path, output_dir)

    print("Extracting any zip archives found within the dataset...")
    extract_archives(output_dir, keep_zips=args.keep_zips)

    print("Dataset ready!")
    print(f"Final dataset location: {output_dir}")


if __name__ == "__main__":
    main()
