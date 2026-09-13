# test_early_stopping.py
# Purpose: guard that patience stops on time and the best epoch's weights are restored.
# Author: Colile Sibanda

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from phishlens.config import load_config
from phishlens.models.text_cnn import TextCNN, build_text_cnn, count_parameters
from phishlens.training.batching import make_dataset, make_loader
from phishlens.training.checkpoint import restore_best, save_checkpoint
from phishlens.training.early_stopping import EarlyStopping
from phishlens.training.epoch_metrics import compute_epoch_metrics
from phishlens.training.loop import CHECKPOINT_FILE, METRIC_FILE, train
from phishlens.training.weighting import positive_class_weight

CONFIG = load_config()


# --------------------------------------------------------------------------
# The patience state machine, on hand-written score sequences.
# --------------------------------------------------------------------------


def test_stops_after_exactly_patience_epochs_without_improvement() -> None:
    """Patience 3: three non-improving epochs stop the run, not two, not four."""
    stopper = EarlyStopping(metric="val_macro_f1", mode="max", patience=3, min_delta=0.0)

    assert stopper.update(0.80) is False  # epoch 1, best
    assert stopper.update(0.70) is False  # epoch 2, 1 without improvement
    assert stopper.update(0.60) is False  # epoch 3, 2 without improvement
    assert stopper.update(0.50) is True   # epoch 4, 3 without improvement -> stop


def test_best_epoch_is_the_improving_one_not_the_last() -> None:
    """The recorded best is the peak epoch, however many worse ones follow."""
    stopper = EarlyStopping(metric="val_macro_f1", mode="max", patience=2, min_delta=0.0)
    for score in (0.70, 0.85, 0.60, 0.55):
        stopper.update(score)

    assert stopper.best_epoch == 2
    assert stopper.best_score == pytest.approx(0.85)


def test_improvement_resets_the_counter() -> None:
    """A better epoch clears the patience counter, so the run continues."""
    stopper = EarlyStopping(metric="val_macro_f1", mode="max", patience=2, min_delta=0.0)

    stopper.update(0.70)
    stopper.update(0.60)
    assert stopper.epochs_without_improvement == 1
    stopper.update(0.90)
    assert stopper.epochs_without_improvement == 0
    assert stopper.best_epoch == 3


def test_improvement_smaller_than_min_delta_does_not_reset() -> None:
    """A gain below min_delta is not an improvement.

    Without this rule a model creeping up by 0.0001 an epoch would train to
    max_epochs while learning nothing worth reporting.
    """
    stopper = EarlyStopping(metric="val_macro_f1", mode="max", patience=2, min_delta=0.01)

    stopper.update(0.900)
    assert stopper.update(0.9005) is False  # below min_delta: counts as no improvement
    assert stopper.epochs_without_improvement == 1
    assert stopper.update(0.9009) is True   # still below: second strike -> stop
    assert stopper.best_epoch == 1
    assert stopper.best_score == pytest.approx(0.900)


def test_min_mode_treats_lower_as_better() -> None:
    """In min mode a falling score is the improvement, for loss-watching runs."""
    stopper = EarlyStopping(metric="val_loss", mode="min", patience=2, min_delta=0.0)

    stopper.update(0.50)
    stopper.update(0.30)
    assert stopper.best_epoch == 2
    assert stopper.update(0.40) is False
    assert stopper.update(0.45) is True
    assert stopper.best_score == pytest.approx(0.30)


def test_rejects_invalid_configuration() -> None:
    """Bad patience, mode or min_delta fail loudly, each naming the argument."""
    with pytest.raises(ValueError, match="mode"):
        EarlyStopping(metric="val_macro_f1", mode="sideways", patience=3, min_delta=0.0)
    with pytest.raises(ValueError, match="patience"):
        EarlyStopping(metric="val_macro_f1", mode="max", patience=0, min_delta=0.0)
    with pytest.raises(ValueError, match="min_delta"):
        EarlyStopping(metric="val_macro_f1", mode="max", patience=3, min_delta=-0.1)


# --------------------------------------------------------------------------
# Checkpoint save and restore.
# --------------------------------------------------------------------------


def test_restore_puts_back_the_saved_weights(tmp_path: Path) -> None:
    """Weights saved, then overwritten, then restored, match the saved ones."""
    model = TextCNN(vocab_size=50, embedding_dim=8, filter_sizes=(3, 4), filters_per_size=4, dropout=0.0)
    path = tmp_path / "best.pt"
    save_checkpoint(model, path)

    saved = model.classifier.weight.detach().clone()
    with torch.no_grad():
        model.classifier.weight.add_(1.0)
    assert not torch.allclose(model.classifier.weight, saved)

    restore_best(model, path)
    assert torch.allclose(model.classifier.weight, saved)


def test_restore_errors_when_the_checkpoint_is_missing(tmp_path: Path) -> None:
    """A missing checkpoint fails loudly rather than leaving the last epoch in place."""
    model = TextCNN(vocab_size=20, embedding_dim=4, filter_sizes=(3,), filters_per_size=2, dropout=0.0)
    with pytest.raises(FileNotFoundError, match="No checkpoint at"):
        restore_best(model, tmp_path / "absent.pt")


# --------------------------------------------------------------------------
# The full loop, on a synthetic dataset with a deliberately worsening score.
# This is the build stated done-when.
# --------------------------------------------------------------------------


class _ScriptedModel(nn.Module):
    """A stand-in model whose validation score follows a fixed script.

    Real training cannot be made to worsen on cue, so the loop is driven by a
    model that returns pre-decided logits: correct for the scripted number of
    epochs, then deliberately wrong. That makes the epoch at which patience
    fires, and the epoch whose weights must come back, both exact rather than
    approximate.

    The `marker` parameter carries the epoch number at the time weights were
    saved, so the restored model can be asked which epoch it came from.

    Inputs: scores — one accuracy target per epoch, in order.
    Output: a module usable by train(); `marker` identifies the epoch.
    """

    def __init__(self, scores: list[float]) -> None:
        super().__init__()
        self.scores = scores
        self.epoch = 0
        self._started = False
        self.marker = nn.Parameter(torch.tensor(0.0))

    def train(self, mode: bool = True) -> "_ScriptedModel":
        """Advance the scripted epoch when the loop switches back to training.

        train() is called once per epoch by the loop, before the training
        pass, so it is the hook that steps the script forward. Validation
        passes call train(False) and must not advance it, or the epoch the
        score belongs to would drift.

        Inputs: mode — True for a training pass, False for evaluation.
        Output: self, matching nn.Module.train.
        """
        if mode and self._started:
            self.epoch += 1
        self._started = self._started or mode
        # Stamp the epoch into the parameter so the weights of epoch 3 and
        # epoch 6 are genuinely different tensors. Without this the model's
        # only parameter never moves, best and last weights are identical,
        # and the restore assertion below would pass even with restore_best
        # deleted — a guard that cannot fail.
        with torch.no_grad():
            self.marker.fill_(float(self.epoch + 1))
        return super().train(mode)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # The first token of each row encodes that row's true label, so the
        # model can be made to agree or disagree with it on demand.
        truth = token_ids[:, 0].float()
        fraction = self.scores[min(self.epoch, len(self.scores) - 1)]

        rows = truth.numel()
        correct = int(round(fraction * rows))
        agreement = torch.zeros(rows)
        agreement[:correct] = 1.0

        # Predict the true label where agreeing, the opposite where not.
        predicted = torch.where(agreement > 0, truth, 1.0 - truth)
        # Keep a real gradient path so the optimiser step is exercised.
        return (predicted * 6.0 - 3.0) + self.marker * 0.0


def _scripted_loaders(config):
    """Build a tiny two-class dataset whose first token encodes the label.

    Inputs: config — Config supplying batch size and max sequence length.
    Output: (loader, labels) with the label recoverable from token 0.
    """
    rows, length = 16, config.features.max_sequence_length
    labels = np.array([1, 0] * (rows // 2), dtype=np.int64)
    sequences = np.full((rows, length), 2, dtype=np.int64)
    sequences[:, 0] = labels

    dataset = make_dataset(sequences, labels)
    return make_loader(dataset, config, shuffle=False), labels


def _tiny_config(tmp_path: Path, max_epochs: int, patience: int):
    """Point paths at tmp_path and set a small epoch budget.

    Inputs: tmp_path — pytest temp directory; max_epochs, patience — loop limits.
    Output: a Config writing its checkpoint and CSV under tmp_path.
    """
    paths = replace(
        CONFIG.paths, models=str(tmp_path / "models"), metrics=str(tmp_path / "metrics")
    )
    early = replace(
        CONFIG.training.early_stopping, patience=patience, min_delta=0.0, metric="val_macro_f1"
    )
    training = replace(
        CONFIG.training, max_epochs=max_epochs, batch_size=8, early_stopping=early
    )
    features = replace(CONFIG.features, max_sequence_length=16)
    return replace(CONFIG, paths=paths, training=training, features=features)


@pytest.fixture()
def worsening_run(tmp_path: Path):
    """Run the loop on a score sequence that peaks at epoch 3 then worsens.

    Scores improve for three epochs, then fall for the rest. With patience 3
    the run must stop at epoch 6 — three non-improving epochs after the peak
    — and must return the epoch-3 weights.

    Inputs: tmp_path — pytest temp directory.
    Output: (TrainingResult, config, model) for the assertions below.
    """
    config = _tiny_config(tmp_path, max_epochs=20, patience=3)
    loader, _ = _scripted_loaders(config)

    # Epochs 1-3 improve, 4 onward worsen.
    model = _ScriptedModel([0.6, 0.8, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    result = train(
        model,
        loader,
        loader,
        config,
        device=torch.device("cpu"),
        verbose=False,
    )
    return result, config, model


def test_loop_stops_after_exactly_patience_epochs(worsening_run) -> None:
    """The run ends at the peak epoch plus patience, not at max_epochs."""
    result, config, _ = worsening_run
    patience = config.training.early_stopping.patience

    assert result.stopped_early is True
    assert result.best_epoch == 3, (
        f"The validation score peaked at epoch 3 but the loop recorded "
        f"{result.best_epoch} as best."
    )
    assert result.epochs_run == result.best_epoch + patience, (
        f"The loop ran {result.epochs_run} epochs; with the peak at "
        f"{result.best_epoch} and patience {patience} it must run exactly "
        f"{result.best_epoch + patience}."
    )
    assert result.epochs_run < config.training.max_epochs, (
        "Early stopping did not fire — the loop ran to max_epochs."
    )


def test_loop_returns_the_best_epochs_weights_not_the_last(worsening_run) -> None:
    """After the run the live model equals the checkpoint, not the final epoch.

    This is the assertion the report depends on: if the restore were skipped,
    every M1 test number would describe a model that was never selected.
    """
    result, _, model = worsening_run

    saved = torch.load(result.checkpoint_path, map_location="cpu", weights_only=True)
    for name, tensor in model.state_dict().items():
        assert torch.allclose(tensor.cpu(), saved[name]), (
            f"Parameter {name!r} differs from the checkpoint, so the model does "
            f"not carry the best epoch's weights."
        )

    # The scripted model stamps its epoch number into `marker`, so the
    # restored weights can be traced to the epoch they came from. Best epoch
    # 3 against 6 epochs run: without the restore this reads 6, which is what
    # makes this assertion able to fail.
    stamped = float(model.marker.detach())
    assert stamped == pytest.approx(float(result.best_epoch)), (
        f"The restored weights are stamped epoch {stamped:.0f} but the "
        f"best epoch was {result.best_epoch}. The model carries the wrong epoch's "
        f"weights, so any test metric reported from it describes a model that was "
        f"never selected."
    )
    assert result.best_epoch != result.epochs_run, (
        "This fixture must end on a non-best epoch, or the restore assertion "
        "above cannot distinguish a working restore from a missing one."
    )


def test_metric_csv_has_one_row_per_epoch(worsening_run) -> None:
    """The per-epoch CSV is the evidence for the metric-logging marks."""
    result, config, _ = worsening_run
    path = Path(config.paths.metrics) / METRIC_FILE

    assert path.is_file(), f"No per-epoch metric CSV at {path}."
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == result.epochs_run, (
        f"The CSV holds {len(rows)} rows for {result.epochs_run} epochs run."
    )
    assert [int(row["epoch"]) for row in rows] == list(range(1, result.epochs_run + 1))

    for column in (
        "train_loss", "train_accuracy", "train_precision", "train_recall",
        "train_f1", "train_macro_f1", "train_auc_roc",
        "val_loss", "val_accuracy", "val_precision", "val_recall",
        "val_f1", "val_macro_f1", "val_auc_roc",
    ):
        assert column in rows[0], f"The per-epoch CSV is missing column {column!r}."


def test_checkpoint_is_written_where_config_says(worsening_run) -> None:
    """The checkpoint lands under paths.models with the configured name."""
    result, config, _ = worsening_run
    assert result.checkpoint_path == Path(config.paths.models) / CHECKPOINT_FILE
    assert result.checkpoint_path.is_file()


def test_loop_rejects_an_unknown_early_stopping_metric(tmp_path: Path) -> None:
    """A metric the loop cannot compute fails before training, naming the options."""
    config = _tiny_config(tmp_path, max_epochs=2, patience=1)
    config = replace(
        config,
        training=replace(
            config.training,
            early_stopping=replace(config.training.early_stopping, metric="val_nonsense"),
        ),
    )
    loader, _ = _scripted_loaders(config)
    model = _ScriptedModel([1.0])

    with pytest.raises(ValueError, match="val_nonsense"):
        train(model, loader, loader, config, device=torch.device("cpu"), verbose=False)


def test_loop_runs_to_max_epochs_when_the_score_keeps_improving(tmp_path: Path) -> None:
    """With no stall, the loop uses its whole budget and restores the last epoch.

    The mirror of the early-stopping test: patience must not fire on a run
    that is still improving.
    """
    config = _tiny_config(tmp_path, max_epochs=4, patience=2)
    loader, _ = _scripted_loaders(config)
    model = _ScriptedModel([0.55, 0.7, 0.85, 1.0])

    result = train(model, loader, loader, config, device=torch.device("cpu"), verbose=False)

    assert result.stopped_early is False
    assert result.epochs_run == 4
    assert result.best_epoch == 4


# --------------------------------------------------------------------------
# The architecture and its supporting pieces.
# --------------------------------------------------------------------------


def test_forward_pass_returns_one_logit_per_row() -> None:
    """A random batch returns shape (batch,), matching the label shape."""
    model = build_text_cnn(vocab_size=500, config=CONFIG)
    batch, length = 4, CONFIG.features.max_sequence_length
    token_ids = torch.randint(0, 500, (batch, length))

    logits = model(token_ids)

    assert logits.shape == (batch,)
    assert logits.dtype == torch.float32
    assert count_parameters(model) > 0


def test_padding_embedding_stays_zero() -> None:
    """The pad index embeds to zero, so padding contributes no learned signal."""
    model = build_text_cnn(vocab_size=100, config=CONFIG)
    assert torch.allclose(model.embedding.weight[0], torch.zeros(CONFIG.model.embedding_dim))


def test_sequence_shorter_than_the_widest_filter_errors() -> None:
    """Too short an input names the filter width rather than crashing in Conv1d."""
    model = TextCNN(vocab_size=50, embedding_dim=8, filter_sizes=(3, 4, 5), filters_per_size=4, dropout=0.0)
    with pytest.raises(ValueError, match="widest filter"):
        model(torch.randint(0, 50, (2, 4)))


def test_rejects_invalid_architecture_arguments() -> None:
    """Each bad hyperparameter fails with its own name in the message."""
    with pytest.raises(ValueError, match="vocab_size"):
        TextCNN(vocab_size=0, embedding_dim=8, filter_sizes=(3,), filters_per_size=4, dropout=0.0)
    with pytest.raises(ValueError, match="dropout"):
        TextCNN(vocab_size=10, embedding_dim=8, filter_sizes=(3,), filters_per_size=4, dropout=1.0)
    with pytest.raises(ValueError, match="filter_sizes"):
        TextCNN(vocab_size=10, embedding_dim=8, filter_sizes=(), filters_per_size=4, dropout=0.0)


def test_positive_class_weight_is_negatives_over_positives() -> None:
    """The weight is computed from training labels alone, by hand-checkable ratio."""
    labels = np.array([1, 1, 0, 0, 0, 0])  # 2 positive, 4 negative
    assert float(positive_class_weight(labels)) == pytest.approx(2.0)


def test_positive_class_weight_rejects_a_single_class() -> None:
    """One class present means no weight is defined; say so rather than divide by zero."""
    with pytest.raises(ValueError, match="only one class"):
        positive_class_weight(np.array([1, 1, 1]))


def test_epoch_metrics_match_a_hand_checked_case() -> None:
    """Precision, recall and F1 on a worked 2x2 case, positive class = 1.

    Labels   1 1 1 0 0 0
    Probs   .9 .8 .2 .7 .1 .1   at threshold 0.5
    Predict  1 1 0 1 0 0
    TP 2, FP 1, FN 1, TN 2 -> precision 2/3, recall 2/3, F1 2/3.
    """
    labels = np.array([1, 1, 1, 0, 0, 0])
    probabilities = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.1])

    metrics = compute_epoch_metrics(labels, probabilities, loss=0.5, threshold=0.5)

    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(2 / 3)
    assert metrics.f1 == pytest.approx(2 / 3)
    assert metrics.accuracy == pytest.approx(4 / 6)
    assert metrics.loss == pytest.approx(0.5)


def test_epoch_metrics_reject_mismatched_arrays() -> None:
    """Labels and probabilities of different lengths cannot be scored together."""
    with pytest.raises(ValueError, match="must describe the same rows"):
        compute_epoch_metrics(np.array([1, 0]), np.array([0.5]), loss=0.1, threshold=0.5)


def test_loader_batch_order_is_reproducible() -> None:
    """Two loaders on the same seed yield the same batches in the same order."""
    config = replace(CONFIG, training=replace(CONFIG.training, batch_size=4))
    sequences = np.arange(32 * 8, dtype=np.int64).reshape(32, 8)
    labels = np.array([1, 0] * 16, dtype=np.int64)

    first = list(make_loader(make_dataset(sequences, labels), config, shuffle=True))
    second = list(make_loader(make_dataset(sequences, labels), config, shuffle=True))

    for (sequences_a, _), (sequences_b, _) in zip(first, second):
        assert torch.equal(sequences_a, sequences_b)
