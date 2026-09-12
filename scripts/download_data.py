# download_data.py
# Purpose: fetch the pooled Kaggle phishing corpus into data/raw/ and verify it.
# Author: Colile Sibanda

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

from phishlens.config.loader import load_config

DATASET_SLUG = "naserabdullahalam/phishing-email-dataset"


def _count_rows(csv_path: Path) -> int:
    """Count data rows in a CSV, excluding the header.

    Inputs: csv_path, path to a corpus CSV file. Email bodies contain
    embedded newlines, so rows are counted via CSV parsing, not by
    counting line breaks.
    Output: the number of data rows.
    """
    return len(pd.read_csv(csv_path))


def download_and_verify(raw_dir: Path, expected_files: tuple[str, ...]) -> dict[str, int]:
    """Download the pooled corpus into raw_dir and verify every expected file.

    Inputs: raw_dir, the target directory; expected_files, the corpus file
    names named in configs/default.yaml.
    Output: a dict mapping file name to row count. Raises FileNotFoundError
    naming the first missing file if the download is partial.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(DATASET_SLUG, path=str(raw_dir), unzip=True, quiet=False)

    row_counts: dict[str, int] = {}
    for file_name in expected_files:
        file_path = raw_dir / file_name
        if not file_path.exists():
            raise FileNotFoundError(
                f"Expected corpus file missing after download: {file_name} "
                f"(looked in {raw_dir})"
            )
        row_counts[file_name] = _count_rows(file_path)
    return row_counts


def main() -> None:
    """Download the six pooled corpora and report per-file row counts."""
    config = load_config()
    raw_dir = Path(config.paths.raw)

    try:
        row_counts = download_and_verify(raw_dir, config.data.corpora)
    except FileNotFoundError as error:
        print(f"Description: download failed — {error}", file=sys.stderr)
        raise

    for file_name, count in row_counts.items():
        print(f"Description: {file_name} row count: {count}")
    print(f"Description: total corpus files verified: {len(row_counts)}")


if __name__ == "__main__":
    main()
