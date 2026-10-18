# early_stopping.py
# Purpose: patience state machine watching the validation metric.
# Author: Colile Sibanda

from __future__ import annotations


class EarlyStopping:
    """Patience counter over a validation score, with min_delta.

    A pure state machine: it sees floats, never tensors or models, so its
    behaviour is unit-testable on a hand-written list of scores, and the
    reported test numbers depend on it being right.

    The rules, stated so they cannot be changed quietly:

    1. An epoch counts as an improvement only if it beats the best score so
       far by at least `min_delta`. An improvement smaller than min_delta does
       not reset the counter and does not become the new best.
    2. The counter resets to zero on every improvement.
    3. `update` returns True when the number of consecutive non-improving
       epochs reaches `patience` : that is, training stops after exactly
       `patience` epochs without improvement.

    Inputs: metric : the name of the watched metric, carried for reporting;
    mode : "max" for metrics where higher is better (macro-F1), "min" for
    metrics where lower is better (loss); patience : consecutive
    non-improving epochs tolerated; min_delta : smallest change counted as an
    improvement.
    Output: a stateful tracker; `update` reports whether to stop, and
    `best_epoch` / `best_score` describe the epoch to restore.

    Raises ValueError naming the offending argument for an unknown mode, a
    non-positive patience or a negative min_delta.
    """

    def __init__(self, metric: str, mode: str, patience: int, min_delta: float) -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"EarlyStopping mode must be 'max' or 'min', got {mode!r}.")
        if patience <= 0:
            raise ValueError(f"EarlyStopping patience must be positive, got {patience}.")
        if min_delta < 0:
            raise ValueError(f"EarlyStopping min_delta must be >= 0, got {min_delta}.")

        self.metric = metric
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta

        self.best_score: float | None = None
        self.best_epoch: int | None = None
        self.epochs_without_improvement = 0
        self._epochs_seen = 0

    def is_improvement(self, score: float) -> bool:
        """Report whether a score beats the best so far by at least min_delta.

        Public because the training loop asks this *before* calling update(),
        to decide whether to write a checkpoint. Both the checkpoint decision
        and the best-epoch record must come from this one rule, or the file on
        disk could stop matching the epoch reported as best.

        Inputs: score : the current epoch's validation score.
        Output: True if this is an improvement under the configured mode.
        """
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta

    def update(self, score: float) -> bool:
        """Record one epoch's score and report whether training should stop.

        Inputs: score : the validation score for the epoch just finished.
        Output: True if `patience` consecutive epochs have passed without an
        improvement, meaning the loop should stop.
        """
        self._epochs_seen += 1

        if self.is_improvement(score):
            self.best_score = score
            self.best_epoch = self._epochs_seen
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        return self.epochs_without_improvement >= self.patience

    def status(self) -> str:
        """Describe the current patience state for the per-epoch log line.

        Inputs: none.
        Output: a human-readable string such as "best 0.9312 at epoch 3,
        2/3 without improvement".
        """
        if self.best_score is None:
            return "no epoch recorded yet"
        return (
            f"best {self.best_score:.4f} at epoch {self.best_epoch}, "
            f"{self.epochs_without_improvement}/{self.patience} without improvement"
        )
