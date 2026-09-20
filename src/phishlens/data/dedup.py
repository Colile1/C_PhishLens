# dedup.py
# Purpose: attach content-hash group ids and report the duplicate rate per corpus.
# Author: Colile Sibanda

from __future__ import annotations

import pandas as pd

from phishlens.data.hashing import content_hash


def assign_groups(frame: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    """Attach a `content_hash` group id to every row.

    Rows are not deleted. Grouping is what prevents leakage : the splitter
    assigns a whole group to one split : whereas deleting near-duplicates
    would discard data without making the split any safer.

    Inputs: frame : cleaned frame; text_column : column to hash.
    Output: a copy of the frame with a `content_hash` column added.

    Raises KeyError if text_column is not present.
    """
    if text_column not in frame.columns:
        raise KeyError(
            f"assign_groups() needs column '{text_column}', which is not in the "
            f"frame. Run clean() first; it produces the 'text' column."
        )
    grouped = frame.copy()
    grouped["content_hash"] = grouped[text_column].map(content_hash)
    return grouped


def duplicate_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarise how much of each corpus is near-duplicate content.

    A row is counted as a duplicate when its content-hash group holds more
    than one row anywhere in the pool, so cross-corpus repetition is counted
    too : that is the repetition a random split would leak.

    Inputs: frame : frame carrying `content_hash` and `corpus` columns.
    Output: DataFrame indexed by corpus with columns rows, unique_groups,
    duplicate_rows, duplicate_rate; plus a final "ALL" row for the pool.

    Raises KeyError if `content_hash` or `corpus` is missing.
    """
    for column in ("content_hash", "corpus"):
        if column not in frame.columns:
            raise KeyError(
                f"duplicate_report() needs column '{column}', which is not in "
                f"the frame. Run assign_groups() after load_corpora()."
            )

    group_sizes = frame["content_hash"].value_counts()
    is_duplicate = frame["content_hash"].map(group_sizes) > 1

    per_corpus = frame.groupby("corpus").agg(
        rows=("content_hash", "size"),
        unique_groups=("content_hash", "nunique"),
    )
    per_corpus["duplicate_rows"] = (
        is_duplicate.groupby(frame["corpus"]).sum().astype(int)
    )
    pooled = pd.DataFrame(
        {
            "rows": [len(frame)],
            "unique_groups": [frame["content_hash"].nunique()],
            "duplicate_rows": [int(is_duplicate.sum())],
        },
        index=pd.Index(["ALL"], name="corpus"),
    )
    report = pd.concat([per_corpus, pooled])
    report["duplicate_rate"] = (report["duplicate_rows"] / report["rows"]).round(4)
    return report
