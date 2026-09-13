# epoch_metrics.py
# Purpose: the per-epoch metric row written to the training CSV.
# Author: Colile Sibanda

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# The positive class is label 1 = phishing/spam, fixed by config
# data.positive_label and asserted here rather than assumed. Every
# precision, recall and F1 below is computed with respect to it.
POSITIVE_LABEL = 1


@dataclass(frozen=True)
class EpochMetrics:
    """One epoch's scores on one split.

    Inputs: loss plus accuracy, precision, recall, F1, macro-F1 and AUC-ROC.
    Output: immutable record; `as_row` flattens it for the CSV.
    """

    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    macro_f1: float
    auc_roc: float

    def as_row(self, prefix: str) -> dict[str, float]:
        """Flatten the metrics into prefixed CSV columns.

        Inputs: prefix — "train" or "val", naming the split the scores describe.
        Output: dict of prefixed column name to value, e.g. "val_macro_f1".
        """
        return {f"{prefix}_{name}": value for name, value in asdict(self).items()}


def compute_epoch_metrics(
    labels: np.ndarray, probabilities: np.ndarray, loss: float, threshold: float
) -> EpochMetrics:
    """Score one epoch's predictions on one split.

    A deliberately minimal helper so the training loop can
    write its per-epoch CSV without pulling the build evaluation layer
    forward onto the wrong step. Step 9 builds `evaluation/metrics.py` with a
    hand-checked confusion matrix and its own critical test; step 10's real M1
    run reports from that module, not this one.

    Inputs: labels — 1-D array of true 0/1 labels; probabilities — 1-D array
    of predicted probabilities for the positive class; loss — the mean loss
    over the split; threshold — probability at or above which a row is
    predicted positive.
    Output: EpochMetrics for the split.

    Raises ValueError if the arrays disagree on length, which would silently
    score predictions against the wrong rows.
    """
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)

    if labels.shape != probabilities.shape:
        raise ValueError(
            f"compute_epoch_metrics() got {labels.shape} labels against "
            f"{probabilities.shape} probabilities. They must describe the same rows."
        )

    predictions = (probabilities >= threshold).astype(int)

    # AUC-ROC is undefined when an epoch's split holds a single class. That
    # happens only on tiny synthetic fixtures, never on the real splits, so it
    # records as NaN rather than failing the run.
    if len(np.unique(labels)) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(labels, probabilities))

    return EpochMetrics(
        loss=float(loss),
        accuracy=float(accuracy_score(labels, predictions)),
        precision=float(
            precision_score(labels, predictions, pos_label=POSITIVE_LABEL, zero_division=0)
        ),
        recall=float(
            recall_score(labels, predictions, pos_label=POSITIVE_LABEL, zero_division=0)
        ),
        f1=float(f1_score(labels, predictions, pos_label=POSITIVE_LABEL, zero_division=0)),
        macro_f1=float(f1_score(labels, predictions, average="macro", zero_division=0)),
        auc_roc=auc,
    )
