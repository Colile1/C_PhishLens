# plots.py
# Purpose: the four Matplotlib figures the brief requires, each returned unsaved.
# Author: Colile Sibanda

from __future__ import annotations

from typing import Sequence

import numpy as np
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

from phishlens.evaluation.metrics import Metrics

# Figure sizes match the ones the build established in notebooks/02_baseline.py,
# so the M0 figures already committed and the M1 figures step 10 produces sit
# side by side in the report at the same scale.
CURVE_FIGSIZE = (6, 5)
MATRIX_FIGSIZE = (5, 5)

# The confusion matrix is labelled in the reader's terms, not the encoding's.
# Label 0 is legitimate, label 1 is phishing/spam, in that order.
CLASS_LABELS = ("legitimate", "phishing")


def _title(figure_name: str, model_name: str, split_name: str) -> str:
    """Build the standard figure title.

    Every figure names what it shows, which model produced it and which split
    it describes, so a reader never has to ask what a figure is. The em-dash
    form matches the M0 figures already committed.

    Inputs: figure_name : what the figure shows; model_name : the model that
    produced it; split_name : the split it was computed on.
    Output: the title string.
    """
    return f"{figure_name} : {model_name}, {split_name} split"


def plot_roc(
    labels: np.ndarray, probabilities: np.ndarray, model_name: str, split_name: str
) -> Figure:
    """Draw the ROC curve with a chance diagonal for reference.

    The figure is returned rather than saved: the notebook displays it inline,
    which is what matters, and decides for itself whether to write a
    file.

    Inputs: labels : 1-D array of true 0/1 labels; probabilities : 1-D array of
    predicted probabilities for the positive class; model_name, split_name :
    named in the title.
    Output: the Matplotlib Figure, with axis labels, a title and a legend.
    """
    figure, axes = plt.subplots(figsize=CURVE_FIGSIZE)
    RocCurveDisplay.from_predictions(labels, probabilities, ax=axes, name=model_name)

    # Without the diagonal a reader cannot see how far above chance the curve
    # sits, which is the only thing the figure is for.
    axes.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    axes.set_xlabel("False positive rate")
    axes.set_ylabel("True positive rate")
    axes.set_title(_title("ROC curve", model_name, split_name))
    axes.legend(loc="lower right")
    figure.tight_layout()
    return figure


def plot_precision_recall(
    labels: np.ndarray, probabilities: np.ndarray, model_name: str, split_name: str
) -> Figure:
    """Draw the precision-recall curve.

    More informative than ROC on an imbalanced positive class, which is why the
    brief asks for both.

    Inputs: labels : 1-D array of true 0/1 labels; probabilities : 1-D array of
    predicted probabilities for the positive class; model_name, split_name :
    named in the title.
    Output: the Matplotlib Figure, with axis labels, a title and a legend.
    """
    figure, axes = plt.subplots(figsize=CURVE_FIGSIZE)
    PrecisionRecallDisplay.from_predictions(
        labels, probabilities, ax=axes, name=model_name
    )

    axes.set_xlabel("Recall")
    axes.set_ylabel("Precision")
    axes.set_title(_title("Precision-recall curve", model_name, split_name))
    axes.legend(loc="lower left")
    figure.tight_layout()
    return figure


def plot_confusion_matrix(metrics: Metrics, model_name: str, split_name: str) -> Figure:
    """Draw the confusion matrix from an already-computed Metrics.

    Takes the Metrics rather than raw arrays on purpose: the figure and the
    numbers quoted beside it in the report then come from one computation at
    one threshold, and cannot disagree. The threshold is named in the title
    because the matrix is meaningless without it.

    Inputs: metrics : the Metrics whose counts to draw; model_name, split_name :
    named in the title.
    Output: the Matplotlib Figure, with axis labels and a title.
    """
    matrix = np.array(
        [
            [metrics.true_negatives, metrics.false_positives],
            [metrics.false_negatives, metrics.true_positives],
        ]
    )

    figure, axes = plt.subplots(figsize=MATRIX_FIGSIZE)
    ConfusionMatrixDisplay(matrix, display_labels=list(CLASS_LABELS)).plot(
        ax=axes, cmap="Blues", colorbar=False
    )

    axes.set_xlabel("Predicted label")
    axes.set_ylabel("True label")
    axes.set_title(
        f"{_title('Confusion matrix', model_name, split_name)}, "
        f"threshold {metrics.threshold:.2f}"
    )
    figure.tight_layout()
    return figure


def plot_loss_curve(history: Sequence[dict[str, float]], model_name: str) -> Figure:
    """Draw training and validation loss against epoch.

    Consumes `TrainingResult.history` from the training loop directly. Two
    series means the legend is not optional : a single unlabelled loss curve
    cannot be read for overfitting, which is the whole reason the brief asks
    for this figure.

    Inputs: history : one flat dict per epoch carrying "epoch", "train_loss"
    and "val_loss", as the training loop produces; model_name : named in the
    title.
    Output: the Matplotlib Figure, with axis labels, a title and a legend.

    Raises ValueError if the history is empty or a required key is absent,
    rather than drawing an empty or half-empty figure.
    """
    if not history:
        raise ValueError(
            "plot_loss_curve() got an empty history. There is nothing to draw; "
            "pass TrainingResult.history from a completed run."
        )

    required = ("epoch", "train_loss", "val_loss")
    missing = [key for key in required if key not in history[0]]
    if missing:
        raise ValueError(
            f"plot_loss_curve() got history rows missing {missing!r}. "
            f"Expected the training loop's per-epoch columns: {', '.join(required)}."
        )

    epochs = [row["epoch"] for row in history]

    figure, axes = plt.subplots(figsize=CURVE_FIGSIZE)
    axes.plot(epochs, [row["train_loss"] for row in history], marker="o", label="Training")
    axes.plot(epochs, [row["val_loss"] for row in history], marker="o", label="Validation")

    axes.set_xlabel("Epoch")
    axes.set_ylabel("Loss")
    axes.set_title(f"Loss curve : {model_name}, training and validation")
    axes.legend()
    figure.tight_layout()
    return figure
