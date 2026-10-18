# loop.py
# Purpose: the explicit training loop with patience and best-weight restore.
# Author: Colile Sibanda

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from phishlens.config.schema import Config
from phishlens.training.checkpoint import restore_best, save_checkpoint
from phishlens.training.early_stopping import EarlyStopping
from phishlens.training.epoch_metrics import EpochMetrics, compute_epoch_metrics

CHECKPOINT_FILE = "text_cnn_best.pt"
METRIC_FILE = "training_log.csv"


@dataclass(frozen=True)
class TrainingResult:
    """What a finished training run reports back.

    Inputs: the epoch whose weights the model now carries, that epoch's
    validation score, the number of epochs actually run, whether early
    stopping fired, the per-epoch history and the checkpoint path.
    Output: immutable summary for the notebook and the report.
    """

    best_epoch: int
    best_score: float
    epochs_run: int
    stopped_early: bool
    history: list[dict[str, float]]
    checkpoint_path: Path


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimiser: torch.optim.Optimizer | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run one pass over a loader, training if an optimiser is supplied.

    One function serves both phases so the forward pass and the probability
    conversion cannot drift apart between training and validation. The
    optimiser argument is what distinguishes them: with one, the pass updates
    weights in train mode; without one, it evaluates under no_grad in eval
    mode, so dropout is disabled.

    Inputs: model : the module; loader : batches of (sequences, labels);
    loss_function : BCEWithLogitsLoss; device : where to run; optimiser : the
    optimiser for a training pass, or None for an evaluation pass.
    Output: (labels, probabilities, mean_loss) for the whole pass.
    """
    training = optimiser is not None
    model.train(training)

    batch_labels: list[np.ndarray] = []
    batch_probabilities: list[np.ndarray] = []
    total_loss = 0.0
    total_rows = 0

    for sequences, labels in loader:
        sequences = sequences.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(training):
            # The model returns logits; the loss applies the sigmoid itself,
            # so probabilities are derived separately and never fed to the loss.
            logits = model(sequences)
            loss = loss_function(logits, labels)

        if training:
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        rows = labels.size(0)
        total_loss += loss.item() * rows
        total_rows += rows
        batch_labels.append(labels.detach().cpu().numpy())
        batch_probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())

    return (
        np.concatenate(batch_labels),
        np.concatenate(batch_probabilities),
        total_loss / total_rows if total_rows else 0.0,
    )


def _write_history(history: list[dict[str, float]], path: Path) -> None:
    """Write the per-epoch metric history to CSV, one row per epoch.

    This file is the per-epoch metric record (accuracy, precision, recall,
    F1, AUC-ROC, loss), so it is rewritten after every epoch rather than only at the end: a run that
    crashes mid-training still leaves the epochs it completed.

    Inputs: history : list of flat per-epoch dicts; path : destination CSV.
    Output: none; writes the CSV with a header row.
    """
    if not history:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Config,
    *,
    positive_weight: torch.Tensor | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> TrainingResult:
    """Train the model with patience-based early stopping and best-weight restore.

    The loop is deliberately explicit and readable: it is printed into the
    training notebook with inspect.getsource so a reader can check the patience
    logic without leaving the notebook, and the patience is implemented in the
    main loop rather than delegated to a framework callback.

    The behaviour that has to be exactly right:

    1. Patience watches `training.early_stopping.metric` : validation
       macro-F1, not loss. Loss and F1 disagree on imbalanced data and the
       report is written about F1.
    2. A checkpoint is written on every improvement, so the file on disk
       always holds the best epoch's weights.
    3. However the loop ends : early stop or max_epochs : the model carries
       the best epoch's weights before this function returns. The test set
       never sees the last epoch unless the last epoch was the best.

    Inputs: model : an unfitted TextCNN; train_loader, val_loader : loaders
    from batching.make_loader; config : validated Config; positive_weight :
    pos_weight for BCEWithLogitsLoss from weighting.positive_class_weight,
    applied only when `training.class_weighting` is set; device : where to
    train, defaulting to CUDA when available; verbose : whether to print a
    line per epoch.
    Output: TrainingResult naming the restored epoch and carrying the history.

    Raises ValueError if the configured early-stopping metric is not a column
    the loop produces, which would otherwise silently watch nothing.
    """
    settings = config.training
    stopping_config = settings.early_stopping
    watched = stopping_config.metric

    # The metric name is validated up front against the columns the loop
    # actually writes, so a typo in the config fails here rather than
    # training for 30 epochs while watching a metric that does not exist.
    available = {f"val_{name}" for name in EpochMetrics.__dataclass_fields__}
    if watched not in available:
        raise ValueError(
            f"Config field 'training.early_stopping.metric' is {watched!r}, which "
            f"the training loop does not compute. Valid options: "
            f"{', '.join(sorted(available))}."
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    weight = positive_weight.to(device) if (settings.class_weighting and positive_weight is not None) else None
    loss_function = nn.BCEWithLogitsLoss(pos_weight=weight)
    optimiser = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)

    stopper = EarlyStopping(
        metric=watched,
        mode=stopping_config.mode,
        patience=stopping_config.patience,
        min_delta=stopping_config.min_delta,
    )

    checkpoint_path = Path(config.paths.models) / CHECKPOINT_FILE
    metric_path = Path(config.paths.metrics) / METRIC_FILE
    threshold = config.evaluation.decision_threshold

    history: list[dict[str, float]] = []
    stopped_early = False

    for epoch in range(1, settings.max_epochs + 1):
        train_labels, train_probabilities, train_loss = _run_epoch(
            model, train_loader, loss_function, device, optimiser
        )
        val_labels, val_probabilities, val_loss = _run_epoch(
            model, val_loader, loss_function, device, None
        )

        train_metrics = compute_epoch_metrics(
            train_labels, train_probabilities, train_loss, threshold
        )
        val_metrics = compute_epoch_metrics(
            val_labels, val_probabilities, val_loss, threshold
        )

        row: dict[str, float] = {"epoch": epoch}
        row.update(train_metrics.as_row("train"))
        row.update(val_metrics.as_row("val"))
        history.append(row)
        _write_history(history, metric_path)

        score = row[watched]
        improved = stopper.is_improvement(score)
        # The checkpoint is written before update() advances the counter, so
        # the file on disk always matches the epoch update() records as best.
        if improved:
            save_checkpoint(model, checkpoint_path)

        should_stop = stopper.update(score)

        if verbose:
            print(
                f"Epoch {epoch}: train_loss {train_loss:.4f}, val_loss {val_loss:.4f}, "
                f"{watched} {score:.4f}, {stopper.status()}"
            )

        if should_stop:
            stopped_early = True
            if verbose:
                print(
                    f"Early stopping: no improvement in {watched} for "
                    f"{stopping_config.patience} epochs. Stopped at epoch {epoch}."
                )
            break

    # Restore before returning, so the object the caller evaluates is the
    # model the reported best score describes.
    if stopping_config.restore_best_weights:
        restore_best(model, checkpoint_path)
        if verbose:
            print(f"Restored weights: epoch {stopper.best_epoch}")

    return TrainingResult(
        best_epoch=stopper.best_epoch if stopper.best_epoch is not None else 0,
        best_score=stopper.best_score if stopper.best_score is not None else float("nan"),
        epochs_run=len(history),
        stopped_early=stopped_early,
        history=history,
        checkpoint_path=checkpoint_path,
    )
