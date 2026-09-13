# __init__.py
# Purpose: public entry point for the training package.
# Author: Colile Sibanda

from phishlens.training.batching import make_dataset, make_loader
from phishlens.training.checkpoint import restore_best, save_checkpoint
from phishlens.training.early_stopping import EarlyStopping
from phishlens.training.epoch_metrics import EpochMetrics, compute_epoch_metrics
from phishlens.training.loop import TrainingResult, train
from phishlens.training.weighting import positive_class_weight

__all__ = [
    "EarlyStopping",
    "EpochMetrics",
    "compute_epoch_metrics",
    "make_dataset",
    "make_loader",
    "positive_class_weight",
    "save_checkpoint",
    "restore_best",
    "train",
    "TrainingResult",
]
