from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from .data_utils import clean_columns


DATA_URL = (
    "https://archive.ics.uci.edu/static/public/165/"
    "concrete%2Bcompressive%2Bstrength.zip"
)


def download_and_prepare(root: Path, force: bool = False) -> Path:
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    output = processed_dir / "concrete_strength.csv"
    if output.exists() and not force:
        print(f"Using existing processed data: {output}")
        return output

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    source = raw_dir / "Concrete_Data.xls"
    if force or not source.exists():
        print(f"Downloading {DATA_URL}")
        with urllib.request.urlopen(DATA_URL, timeout=60) as response:
            archive_bytes = response.read()
        archive_path = raw_dir / "concrete_compressive_strength.zip"
        archive_path.write_bytes(archive_bytes)
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            archive.extractall(raw_dir)
    else:
        print(f"Using existing raw data: {source}")
    frame = clean_columns(pd.read_excel(source))
    frame.to_csv(output, index=False)
    print(f"Saved {len(frame)} observations to {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare the UCI dataset")
    parser.add_argument("--force", action="store_true", help="redownload existing data")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    download_and_prepare(root, force=args.force)


if __name__ == "__main__":
    main()

