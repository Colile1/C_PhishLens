# metrics.py
# Purpose: the reported metric set, computed from probabilities at an explicit threshold.
# Author: Colile Sibanda

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

# The positive class is label 1 = phishing/spam, fixed by config
# data.positive_label. Every precision, recall and F1 below is computed with
# respect to it. This is checked against the data rather than assumed: a
# corpus encoded -1/1 or 1/2 would otherwise score silently and wrongly, and
# the number would go straight into the report.
POSITIVE_LABEL = 1

# The metric names that both this module and training/epoch_metrics.py
# compute. The two are deliberately separate modules (see the docstring on
# compute_epoch_metrics) and this tuple is what the cross-check test in
# tests/critical/test_metric_correctness.py uses to prove they have not
# drifted apart.
SHARED_METRIC_NAMES = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "macro_f1",
    "auc_roc",
)

_OBJECTIVES = ("macro_f1", "f1")


@dataclass(frozen=True)
class Metrics:
    """The full metric set for one model on one split.

    Carries the threshold it was computed at, so a Metrics is self-describing
    and cannot be reported beside the wrong operating point. The confusion
    matrix is stored as four named counts rather than a 2x2 array because the
    false-positive review asks for false_positives by name,
    and because a named field cannot be transposed by accident.

    Inputs: the seven scores, the four confusion-matrix counts, and the
    decision threshold the thresholded scores used.
    Output: immutable record; `as_dict` returns the scores alone.
    """

    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    macro_f1: float
    auc_roc: float
    average_precision: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int

    def as_dict(self) -> dict[str, float]:
        """Return the metric scores alone, without the threshold or the counts.

        This is what `persistence.save_metrics` nests under "metrics" and what
        `reporting.metrics_table` renders as a row, so it holds scores only :
        the counts are integers describing the matrix, not scores to compare
        between models.

        Inputs: none.
        Output: dict of metric name to value, in declaration order.
        """
        skip = {
            "threshold",
            "true_negatives",
            "false_positives",
            "false_negatives",
            "true_positives",
        }
        return {
            name: value for name, value in asdict(self).items() if name not in skip
        }


def _check_arrays(labels: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate the label and probability arrays before anything is scored.

    Both failures this catches are silent ones. Mismatched lengths would score
    predictions against the wrong rows. A label set outside {0, 1} would make
    `pos_label=1` select the wrong class, or select nothing, and still return
    a plausible-looking number.

    Inputs: labels : 1-D array of true labels; probabilities : 1-D array of
    predicted probabilities for the positive class.
    Output: (labels, probabilities) as numpy arrays.

    Raises ValueError if the shapes disagree, or if the labels are not drawn
    from {0, 1}.
    """
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)

    if labels.shape != probabilities.shape:
        raise ValueError(
            f"compute_metrics() got {labels.shape} labels against "
            f"{probabilities.shape} probabilities. They must describe the same rows."
        )

    present = set(np.unique(labels).tolist())
    if not present <= {0, 1}:
        raise ValueError(
            f"compute_metrics() got labels {sorted(present)!r}. The positive "
            f"class is {POSITIVE_LABEL!r} and labels must be drawn from "
            f"{{0, 1}}, so a different encoding would score the wrong class."
        )

    return labels, probabilities


def compute_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> Metrics:
    """Score predictions against labels at an explicit decision threshold.

    Takes probabilities rather than hard labels so AUC-ROC and average
    precision are computable and the threshold stays visible in the signature.
    A row is predicted positive when its probability is at or above the
    threshold, matching training/epoch_metrics.py : `>` instead of `>=` would
    move every boundary case by one row.

    Inputs: labels : 1-D array of true 0/1 labels; probabilities : 1-D array of
    predicted probabilities for the positive class (for a sklearn classifier
    that is `predict_proba(...)[:, 1]`, not column 0); threshold : probability
    at or above which a row is predicted positive.
    Output: Metrics for the split, carrying the threshold used.

    Raises ValueError if the arrays disagree on length or the labels are not
    drawn from {0, 1}.
    """
    labels, probabilities = _check_arrays(labels, probabilities)
    predictions = (probabilities >= threshold).astype(int)

    # Pinning labels=[0, 1] fixes the matrix as [[TN, FP], [FN, TP]] even when
    # a split or a fixture happens to hold a single class, where sklearn would
    # otherwise infer a 1x1 matrix and ravel() would raise.
    true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()

    # AUC-ROC and average precision are undefined when the split holds a single
    # class. That happens only on small fixtures, never on the real splits, so
    # they record as NaN rather than failing the run.
    if len(np.unique(labels)) < 2:
        auc = float("nan")
        average_precision = float("nan")
    else:
        auc = float(roc_auc_score(labels, probabilities))
        average_precision = float(average_precision_score(labels, probabilities))

    return Metrics(
        threshold=float(threshold),
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
        average_precision=average_precision,
        true_negatives=int(true_negatives),
        false_positives=int(false_positives),
        false_negatives=int(false_negatives),
        true_positives=int(true_positives),
    )


def select_threshold(
    labels: np.ndarray, probabilities: np.ndarray, objective: str = "macro_f1"
) -> float:
    """Choose the decision threshold that maximises an objective.

    **Call this on the validation split only.** Selecting a threshold against
    the test split would spend the project's once-only test budget and turn a
    threshold chosen here is then applied unchanged to test,
    alongside the untuned 0.5 figure.

    Candidates come from the precision-recall curve, which changes only at the
    distinct probability values present, so sweeping a fixed grid would test
    many thresholds that produce identical predictions while missing the ones
    that matter.

    Inputs: labels : 1-D array of true 0/1 labels; probabilities : 1-D array of
    predicted probabilities for the positive class; objective : "macro_f1" or
    "f1", the score to maximise.
    Output: the threshold maximising the objective. Ties resolve to the lowest
    such threshold, which is the higher-recall choice.

    Raises ValueError if the arrays are invalid, if the objective is unknown,
    or if the labels hold a single class, where no operating point is defined.
    """
    labels, probabilities = _check_arrays(labels, probabilities)

    if objective not in _OBJECTIVES:
        raise ValueError(
            f"select_threshold() got objective {objective!r}. "
            f"Valid options: {', '.join(_OBJECTIVES)}."
        )

    if len(np.unique(labels)) < 2:
        raise ValueError(
            "select_threshold() needs both classes present to define an "
            "operating point, but the labels hold a single class."
        )

    # precision_recall_curve returns one fewer threshold than it does precision
    # and recall values, the final point being recall 0 at no threshold.
    _, _, candidates = precision_recall_curve(labels, probabilities)

    best_threshold = float(candidates[0])
    best_score = -1.0
    for candidate in candidates:
        scored = compute_metrics(labels, probabilities, float(candidate))
        score = getattr(scored, objective)
        if score > best_score:
            best_score = score
            best_threshold = float(candidate)

    return best_threshold
