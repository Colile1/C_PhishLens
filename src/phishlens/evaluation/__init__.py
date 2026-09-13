# __init__.py
# Purpose: public entry point for the evaluation package.
# Author: Colile Sibanda

from phishlens.evaluation.metrics import (
    POSITIVE_LABEL,
    Metrics,
    compute_metrics,
    select_threshold,
)
from phishlens.evaluation.persistence import commit_hash, save_metrics
from phishlens.evaluation.plots import (
    plot_confusion_matrix,
    plot_loss_curve,
    plot_precision_recall,
    plot_roc,
)
from phishlens.evaluation.reporting import metrics_table, per_corpus_table

__all__ = [
    "Metrics",
    "compute_metrics",
    "select_threshold",
    "POSITIVE_LABEL",
    "plot_roc",
    "plot_precision_recall",
    "plot_confusion_matrix",
    "plot_loss_curve",
    "metrics_table",
    "per_corpus_table",
    "save_metrics",
    "commit_hash",
]
