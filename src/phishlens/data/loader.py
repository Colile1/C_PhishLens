# loader.py
# Purpose: read the six raw corpus CSVs and pool them into one tagged frame.
# Author: Colile Sibanda

from __future__ import annotations

from pathlib import Path

import pandas as pd

from phishlens.config.schema import Config

# Columns every corpus must provide. The four audit-only columns
# (sender, receiver, date, urls) are present in only some corpora and are
# filled with NA where absent, so the pooled frame has one stable shape.
REQUIRED_COLUMNS = ("subject", "body", "label")


def corpus_name_for(file_name: str) -> str:
    """Derive the corpus tag from a corpus file name.

    Inputs: file_name : a corpus file name such as "CEAS_08.csv".
    Output: the file's stem, used as the value of the `corpus` column.
    """
    return Path(file_name).stem


def load_corpora(config: Config) -> pd.DataFrame:
    """Read every configured corpus CSV and concatenate them into one frame.

    Each corpus contributes a `corpus` column naming its source, so the
    corpus probe and the leave-one-corpus-out evaluation can recover it.
    Column names are harmonised: the required text and label columns must be
    present in every corpus, and the audit-only columns are added as NA where
    a corpus does not carry them.

    Inputs: config : validated Config; reads `config.data.corpora` from
    `config.paths.raw`.
    Output: pooled DataFrame with the required columns, the audit-only
    columns, and `corpus`, indexed 0..n-1.

    Raises FileNotFoundError naming every missing corpus file, and KeyError
    naming the corpus and the columns it is missing.
    """
    raw_dir = Path(config.paths.raw)

    missing_files = [
        file_name
        for file_name in config.data.corpora
        if not (raw_dir / file_name).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"Missing corpus files in {raw_dir}: {', '.join(missing_files)}. "
            "Run scripts/download_data.py to fetch the dataset."
        )

    audit_columns = list(config.data.audit_only_columns)
    keep_columns = list(REQUIRED_COLUMNS) + audit_columns

    frames = []
    for file_name in config.data.corpora:
        frame = pd.read_csv(raw_dir / file_name)
        absent_required = [
            column for column in REQUIRED_COLUMNS if column not in frame.columns
        ]
        if absent_required:
            raise KeyError(
                f"Corpus {file_name} is missing required column(s): "
                f"{', '.join(absent_required)}. Expected all of "
                f"{', '.join(REQUIRED_COLUMNS)}."
            )
        for column in audit_columns:
            if column not in frame.columns:
                frame[column] = pd.NA
        frame = frame[keep_columns].copy()
        frame["corpus"] = corpus_name_for(file_name)
        frames.append(frame)

    pooled = pd.concat(frames, ignore_index=True)
    return pooled
