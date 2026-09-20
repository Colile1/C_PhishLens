# loco.py
# Purpose: leave-one-corpus-out train/test indices for the generalisation check.
# Author: Colile Sibanda

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class LocoFold:
    """One leave-one-corpus-out fold: train on five corpora, test on the sixth.

    Inputs: held_out corpus name, and the content-hash group ids for each side.
    Output: immutable fold description.
    """

    held_out: str
    train: frozenset[str]
    test: frozenset[str]


def make_loco_index(frame: pd.DataFrame) -> dict[str, LocoFold]:
    """Build one train/test fold per source corpus.

    The held-out corpus is the test side and the other five are the train
    side. Groups that appear in both : the same email content present in two
    corpora, which the pooled corpora do contain : are removed from the train
    side, so the held-out corpus stays genuinely unseen. Without that removal
    LOCO would quietly measure the same leakage the random split is guarded
    against, and the gap between the two would be meaningless.

    Inputs: frame : cleaned frame carrying `content_hash` and `corpus`.
    Output: dict of corpus name to LocoFold, ordered by corpus name.

    Raises KeyError naming any required column that is absent.
    """
    for column in ("content_hash", "corpus"):
        if column not in frame.columns:
            raise KeyError(
                f"make_loco_index() needs column '{column}', which is not in "
                f"the frame. Run clean() then assign_groups() first."
            )

    folds: dict[str, LocoFold] = {}
    for corpus in sorted(frame["corpus"].unique()):
        is_held_out = frame["corpus"] == corpus
        test_groups = frozenset(frame.loc[is_held_out, "content_hash"])
        train_groups = frozenset(frame.loc[~is_held_out, "content_hash"]) - test_groups
        folds[corpus] = LocoFold(
            held_out=corpus, train=train_groups, test=test_groups
        )
    return folds


def loco_report(
    frame: pd.DataFrame,
    folds: dict[str, LocoFold],
    label_column: str = "label",
) -> pd.DataFrame:
    """Summarise the size and class balance of each leave-one-corpus-out fold.

    Inputs: frame : the frame the folds were built from; folds : output of
    `make_loco_index`; label_column : the label column name from config.
    Output: DataFrame indexed by held-out corpus with train/test row counts,
    the test-side positive rate, and the count of groups shared with the
    train side before removal.
    """
    rows = []
    for corpus, fold in folds.items():
        is_held_out = frame["corpus"] == corpus
        other_groups = frozenset(frame.loc[~is_held_out, "content_hash"])
        in_train = frame["content_hash"].isin(fold.train)
        in_test = frame["content_hash"].isin(fold.test)
        rows.append(
            {
                "held_out": corpus,
                "train_rows": int(in_train.sum()),
                "test_rows": int(in_test.sum()),
                "test_positive_rate": round(
                    float(frame.loc[in_test, label_column].mean()), 4
                )
                if in_test.any()
                else 0.0,
                "shared_groups_removed": len(fold.test & other_groups),
            }
        )
    return pd.DataFrame(rows).set_index("held_out")
