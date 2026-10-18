# reporting.py
# Purpose: render computed metrics as the comparison tables the report quotes.
# Author: Colile Sibanda

from __future__ import annotations

import pandas as pd

from phishlens.evaluation.metrics import Metrics

# Column order for the comparison tables. Fixed here rather than taken from
# dict order so the M0 vs M1 table and the per-corpus table read the same way
# round, and so a reordered dataclass cannot quietly reshuffle the report.
TABLE_COLUMNS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "macro_f1",
    "auc_roc",
    "average_precision",
)

DISPLAY_PRECISION = 4


def _frame(results: dict[str, Metrics], index_name: str) -> pd.DataFrame:
    """Build a metrics DataFrame from a name-to-Metrics mapping.

    Inputs: results : mapping of row name to its Metrics; index_name : what the
    index column is called, "model" or "corpus".
    Output: DataFrame indexed by name, one column per metric in TABLE_COLUMNS.

    Raises ValueError if results is empty, which would render as a blank table
    in the report rather than announcing that nothing was passed.
    """
    if not results:
        raise ValueError(
            f"No results to tabulate. Pass a mapping of {index_name} name to "
            f"Metrics with at least one entry."
        )

    frame = pd.DataFrame(
        {name: metrics.as_dict() for name, metrics in results.items()}
    ).T
    frame = frame[list(TABLE_COLUMNS)]
    frame.index.name = index_name
    return frame.round(DISPLAY_PRECISION)


def metrics_table(results: dict[str, Metrics]) -> pd.DataFrame:
    """Tabulate several models side by side for the M0 vs M1 comparison.

    Formatting only : every value comes from an already-computed Metrics, so
    this cannot introduce a number that was not measured.

    Inputs: results : mapping of model name to its Metrics, e.g.
    {"M0 TF-IDF + LogReg": ..., "M1 Text-CNN": ...}.
    Output: DataFrame indexed by model name, one column per metric.

    Raises ValueError if results is empty.
    """
    return _frame(results, index_name="model")


def per_corpus_table(results: dict[str, Metrics]) -> pd.DataFrame:
    """Tabulate one model's scores broken down by corpus.

    Indexed by corpus name to match `data.loco.loco_report`, so the
    leave-one-corpus-out results (notebooks/05_loco.ipynb) and the per-corpus
    breakdown (notebooks/04_evaluate.ipynb) line up row for row in the report.

    Inputs: results : mapping of corpus name to that corpus's Metrics.
    Output: DataFrame indexed by corpus name, sorted by name for a stable
    reading order, one column per metric.

    Raises ValueError if results is empty.
    """
    return _frame(results, index_name="corpus").sort_index()
