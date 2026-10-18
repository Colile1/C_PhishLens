# test_metric_correctness.py
# Purpose: guard that reported metrics match hand-computed values and score the right class.
# Author: Colile Sibanda

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import pytest

# Agg renders figures without a display, so the figure tests run headless in
# CI and on a machine with no GUI backend available.
matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402

from phishlens.evaluation.metrics import (  # noqa: E402
    POSITIVE_LABEL,
    SHARED_METRIC_NAMES,
    Metrics,
    compute_metrics,
    select_threshold,
)
from phishlens.evaluation.persistence import (  # noqa: E402
    UNKNOWN_COMMIT,
    commit_hash,
    save_metrics,
)
from phishlens.evaluation.plots import (  # noqa: E402
    plot_confusion_matrix,
    plot_loss_curve,
    plot_precision_recall,
    plot_roc,
)
from phishlens.evaluation.reporting import metrics_table, per_corpus_table  # noqa: E402
from phishlens.training.epoch_metrics import compute_epoch_metrics  # noqa: E402

# Six decimal places, as the folder README requires. Hand-computed values are
# exact fractions, so anything looser would hide a real arithmetic error.
TOLERANCE = 1e-6


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every figure after each test so the suite does not leak them.

    Matplotlib keeps created figures alive in a global registry. Without this
    a full run accumulates them and warns about too many open figures.
    """
    yield
    plt.close("all")


# --------------------------------------------------------------------------
# The hand-checked confusion matrix: every metric is derived from these counts.
# --------------------------------------------------------------------------


# Ten rows, worked out by hand. Deliberately asymmetric: TP, FP, FN and TN are
# four different numbers and precision != recall, so a transposed pair or a
# swapped count fails. The symmetric case already in test_early_stopping.py
# (TP 2, FP 1, FN 1, TN 2 -> 2/3, 2/3, 2/3) cannot catch either error, which
# is why this one is different rather than a copy.
#
#   labels  1    1    1    1    0    0    0    0    0    0
#   probs  .95  .80  .60  .30  .90  .55  .40  .20  .10  .05
#   pred    1    1    1    0    1    1    0    0    0    0     at threshold 0.5
#           TP   TP   TP   FN   FP   FP   TN   TN   TN   TN
#
# TP 3, FP 2, FN 1, TN 4
#   precision = 3 / (3 + 2) = 0.6
#   recall    = 3 / (3 + 1) = 0.75
#   F1        = 2 * 0.6 * 0.75 / (0.6 + 0.75) = 0.9 / 1.35 = 2/3
#   accuracy  = (3 + 4) / 10 = 0.7
#
# Negative class, for the macro average:
#   precision = 4 / (4 + 1) = 0.8
#   recall    = 4 / (4 + 2) = 2/3
#   F1        = 2 * 0.8 * (2/3) / (0.8 + 2/3) = (16/15) / (22/15) = 8/11
#   macro F1  = (2/3 + 8/11) / 2 = (22/33 + 24/33) / 2 = (46/33) / 2 = 23/33
HAND_LABELS = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
HAND_PROBABILITIES = np.array([0.95, 0.80, 0.60, 0.30, 0.90, 0.55, 0.40, 0.20, 0.10, 0.05])


@pytest.fixture()
def hand_metrics() -> Metrics:
    """Score the hand-worked case at threshold 0.5.

    Inputs: none.
    Output: the Metrics for the matrix documented above.
    """
    return compute_metrics(HAND_LABELS, HAND_PROBABILITIES, threshold=0.5)


def test_confusion_matrix_counts_match_the_hand_worked_case(hand_metrics) -> None:
    """The four counts are TP 3, FP 2, FN 1, TN 4, each in its own field."""
    assert hand_metrics.true_positives == 3
    assert hand_metrics.false_positives == 2
    assert hand_metrics.false_negatives == 1
    assert hand_metrics.true_negatives == 4

    total = (
        hand_metrics.true_positives
        + hand_metrics.false_positives
        + hand_metrics.false_negatives
        + hand_metrics.true_negatives
    )
    assert total == len(HAND_LABELS), (
        f"The counts sum to {total} for {len(HAND_LABELS)} rows, so at least "
        f"one row was counted twice or not at all."
    )


def test_precision_recall_and_f1_match_hand_computed_values(hand_metrics) -> None:
    """Precision 0.6, recall 0.75, F1 2/3 : to six decimal places.

    The three differ from each other, so this fails if precision and recall are
    transposed, which a symmetric fixture could not detect.
    """
    assert hand_metrics.precision == pytest.approx(0.6, abs=TOLERANCE)
    assert hand_metrics.recall == pytest.approx(0.75, abs=TOLERANCE)
    assert hand_metrics.f1 == pytest.approx(2 / 3, abs=TOLERANCE)
    assert hand_metrics.accuracy == pytest.approx(0.7, abs=TOLERANCE)
    assert hand_metrics.macro_f1 == pytest.approx(23 / 33, abs=TOLERANCE)

    assert hand_metrics.precision != pytest.approx(hand_metrics.recall, abs=TOLERANCE), (
        "This fixture must keep precision and recall distinct, or a transposition "
        "of the two would pass unnoticed."
    )


def test_auc_matches_a_hand_computed_value(hand_metrics) -> None:
    """AUC-ROC is 19/24 by counting concordant positive/negative pairs.

    AUC equals the probability that a random positive outranks a random
    negative. With 4 positives and 6 negatives there are 24 such pairs. The
    positive probabilities are .95 .80 .60 .30 and the negatives are
    .90 .55 .40 .20 .10 .05, with no ties between the two groups.

    Counting, for each positive, the negatives it outranks:
      .95 beats all 6
      .80 beats .55 .40 .20 .10 .05           = 5   (loses to .90)
      .60 beats .55 .40 .20 .10 .05           = 5   (loses to .90)
      .30 beats .20 .10 .05                   = 3   (loses to .90 .55 .40)
    Total 6 + 5 + 5 + 3 = 19, so AUC = 19/24 = 0.791666...
    """
    expected = 19 / 24
    assert hand_metrics.auc_roc == pytest.approx(expected, abs=TOLERANCE)


def test_threshold_is_recorded_on_the_metrics(hand_metrics) -> None:
    """The Metrics carries the threshold it was computed at.

    Without this a tuned operating point could be reported under a 0.5 heading,
    which is exactly what D8's two-threshold reporting risks.
    """
    assert hand_metrics.threshold == pytest.approx(0.5)
    assert compute_metrics(HAND_LABELS, HAND_PROBABILITIES, 0.8).threshold == pytest.approx(0.8)


# --------------------------------------------------------------------------
# The positive class. A guard that cannot fail is not a guard, so these test
# the consequence of the positive class being 1, not merely the constant.
# --------------------------------------------------------------------------


def test_positive_class_is_label_one() -> None:
    """The module scores label 1 as positive, per config data.positive_label."""
    assert POSITIVE_LABEL == 1


def test_metrics_describe_label_one_not_label_zero(hand_metrics) -> None:
    """Precision and recall are the label-1 values, not the label-0 ones.

    This is the assertion with teeth. On the hand-worked case label 1 scores
    precision 0.6 / recall 0.75 while label 0 scores precision 0.8 / recall
    2/3. If the positive class were flipped, every value below would still look
    entirely plausible : and every phishing number in the report would describe
    the legitimate class instead.
    """
    label_zero_precision = 4 / 5
    label_zero_recall = 4 / 6

    assert hand_metrics.precision == pytest.approx(0.6, abs=TOLERANCE), (
        f"Precision is {hand_metrics.precision:.6f}. The label-1 value is 0.6 and "
        f"the label-0 value is {label_zero_precision:.6f} : this reads as the "
        f"wrong class."
    )
    assert hand_metrics.precision != pytest.approx(label_zero_precision, abs=TOLERANCE)
    assert hand_metrics.recall != pytest.approx(label_zero_recall, abs=TOLERANCE)


def test_labels_outside_zero_and_one_are_rejected() -> None:
    """A -1/1 or 1/2 encoding fails loudly rather than scoring the wrong class."""
    probabilities = np.array([0.9, 0.2, 0.8])

    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        compute_metrics(np.array([-1, 1, 1]), probabilities, threshold=0.5)

    with pytest.raises(ValueError, match="drawn from"):
        compute_metrics(np.array([1, 2, 2]), probabilities, threshold=0.5)


def test_mismatched_arrays_are_rejected() -> None:
    """Labels and probabilities of different lengths cannot be scored together."""
    with pytest.raises(ValueError, match="must describe the same rows"):
        compute_metrics(np.array([1, 0]), np.array([0.5]), threshold=0.5)


def test_the_wrong_probability_column_is_visible_in_auc() -> None:
    """Passing 1 - p, the negative-class column, drops AUC below chance.

    The folder README names this as one of the two silent failure modes:
    predict_proba(...)[:, 0] instead of [:, 1] returns a well-formed array and
    plausible accuracy, but inverts the ranking. AUC is where it shows.
    """
    correct = compute_metrics(HAND_LABELS, HAND_PROBABILITIES, threshold=0.5)
    flipped = compute_metrics(HAND_LABELS, 1.0 - HAND_PROBABILITIES, threshold=0.5)

    assert correct.auc_roc > 0.5
    assert flipped.auc_roc < 0.5
    assert flipped.auc_roc == pytest.approx(1.0 - correct.auc_roc, abs=TOLERANCE)


# --------------------------------------------------------------------------
# Threshold behaviour, including the >= boundary and the design selection helper.
# --------------------------------------------------------------------------


def test_threshold_comparison_includes_the_boundary() -> None:
    """A probability exactly at the threshold predicts positive, not negative.

    The loop and this module must agree on >= rather than >, or a row sitting
    exactly on the threshold is classified differently in training and in the
    report.
    """
    labels = np.array([1, 0])
    probabilities = np.array([0.5, 0.5])

    scored = compute_metrics(labels, probabilities, threshold=0.5)

    assert scored.true_positives == 1, "A probability equal to the threshold must predict positive."
    assert scored.false_positives == 1
    assert scored.false_negatives == 0


def test_raising_the_threshold_trades_recall_for_precision() -> None:
    """Moving the threshold trades recall away for precision.

    At 0.25 every positive is caught (recall 1.0) at the cost of 3 false
    positives (precision 4/7). At 0.95 only the top-ranked row is predicted
    positive: precision 1.0, recall 0.25.

    Note the trade is not monotonic on this fixture : at 0.85 precision is 0.5,
    *below* the 0.25 value, because the highest-scoring row overall is a
    negative (.90) and it survives into a much smaller positive set. That is a
    real property of the data, not a defect, and it is why a threshold is
    chosen by searching the curve rather than by assuming higher is stricter.
    """
    low = compute_metrics(HAND_LABELS, HAND_PROBABILITIES, threshold=0.25)
    high = compute_metrics(HAND_LABELS, HAND_PROBABILITIES, threshold=0.95)

    assert low.recall == pytest.approx(1.0, abs=TOLERANCE)
    assert low.precision == pytest.approx(4 / 7, abs=TOLERANCE)

    assert high.precision == pytest.approx(1.0, abs=TOLERANCE)
    assert high.recall == pytest.approx(0.25, abs=TOLERANCE)

    assert low.recall > high.recall
    assert high.precision > low.precision


def test_select_threshold_finds_the_macro_f1_optimum() -> None:
    """On a separable case the chosen threshold scores at least the 0.5 default.

    The positives sit above 0.6 and the negatives below 0.45, so a threshold
    between them classifies every row correctly and macro-F1 reaches 1.0.
    """
    labels = np.array([1, 1, 1, 0, 0, 0])
    probabilities = np.array([0.95, 0.80, 0.65, 0.44, 0.20, 0.05])

    chosen = select_threshold(labels, probabilities, objective="macro_f1")
    chosen_metrics = compute_metrics(labels, probabilities, chosen)
    default_metrics = compute_metrics(labels, probabilities, 0.5)

    assert chosen_metrics.macro_f1 == pytest.approx(1.0, abs=TOLERANCE)
    assert chosen_metrics.macro_f1 >= default_metrics.macro_f1
    assert 0.44 < chosen <= 0.65


def test_select_threshold_rejects_an_unknown_objective() -> None:
    """An objective the helper cannot maximise fails, naming the valid options."""
    with pytest.raises(ValueError, match="nonsense"):
        select_threshold(HAND_LABELS, HAND_PROBABILITIES, objective="nonsense")


def test_select_threshold_needs_both_classes() -> None:
    """One class present means no operating point is defined; say so."""
    with pytest.raises(ValueError, match="single class"):
        select_threshold(np.array([1, 1, 1]), np.array([0.9, 0.8, 0.7]))


def test_single_class_metrics_record_nan_rather_than_failing() -> None:
    """AUC and average precision are undefined on one class, and record as NaN."""
    scored = compute_metrics(np.array([1, 1]), np.array([0.9, 0.8]), threshold=0.5)

    assert math.isnan(scored.auc_roc)
    assert math.isnan(scored.average_precision)
    assert scored.true_positives == 2


# --------------------------------------------------------------------------
# The cross-check that makes the duplication with training/epoch_metrics.py
# safe. The two modules are separate by design; this proves they agree.
# --------------------------------------------------------------------------


def test_agrees_with_the_per_epoch_helper_on_every_shared_metric() -> None:
    """compute_metrics and compute_epoch_metrics score identically.

    training/epoch_metrics.py deliberately duplicates part of this module so
    the training layer does not depend on the evaluation layer. That duplication is only
    safe while the two agree: if either drifts, the per-epoch CSV and the
    reported headline numbers would describe the same model differently.
    """
    reported = compute_metrics(HAND_LABELS, HAND_PROBABILITIES, threshold=0.5)
    per_epoch = compute_epoch_metrics(
        HAND_LABELS, HAND_PROBABILITIES, loss=0.0, threshold=0.5
    )

    for name in SHARED_METRIC_NAMES:
        assert getattr(reported, name) == pytest.approx(
            getattr(per_epoch, name), abs=TOLERANCE
        ), (
            f"evaluation.metrics and training.epoch_metrics disagree on {name!r}: "
            f"{getattr(reported, name)} against {getattr(per_epoch, name)}."
        )


# --------------------------------------------------------------------------
# The four required figures. Each must be readable without asking what it is.
# --------------------------------------------------------------------------


def _history() -> list[dict[str, float]]:
    """Build a per-epoch history in the shape the training loop returns.

    Inputs: none.
    Output: three epochs carrying the loop's epoch/train_loss/val_loss columns.
    """
    return [
        {"epoch": 1, "train_loss": 0.70, "val_loss": 0.72},
        {"epoch": 2, "train_loss": 0.45, "val_loss": 0.50},
        {"epoch": 3, "train_loss": 0.30, "val_loss": 0.48},
    ]


def test_all_four_figures_carry_a_title_and_labelled_axes(hand_metrics) -> None:
    """Every required figure is self-describing. This is the README's rule.

    A figure in the report with an unlabelled axis misleads and cannot be
    interpreted, so the labels are asserted rather than left to discipline.
    """
    figures = {
        "roc": plot_roc(HAND_LABELS, HAND_PROBABILITIES, "M1 Text-CNN", "validation"),
        "precision_recall": plot_precision_recall(
            HAND_LABELS, HAND_PROBABILITIES, "M1 Text-CNN", "validation"
        ),
        "confusion_matrix": plot_confusion_matrix(
            hand_metrics, "M1 Text-CNN", "validation"
        ),
        "loss_curve": plot_loss_curve(_history(), "M1 Text-CNN"),
    }

    for name, figure in figures.items():
        axes = figure.axes[0]
        assert axes.get_title().strip(), f"The {name} figure has no title."
        assert axes.get_xlabel().strip(), f"The {name} figure has no x-axis label."
        assert axes.get_ylabel().strip(), f"The {name} figure has no y-axis label."


def test_roc_and_loss_curves_carry_a_legend() -> None:
    """Figures drawing more than one series name them.

    The loss curve is the case that matters: two unlabelled lines cannot be
    read for overfitting, which is the only reason the figure is required.
    """
    roc = plot_roc(HAND_LABELS, HAND_PROBABILITIES, "M1 Text-CNN", "validation")
    assert roc.axes[0].get_legend() is not None

    loss = plot_loss_curve(_history(), "M1 Text-CNN")
    axes = loss.axes[0]
    assert axes.get_legend() is not None
    assert len(axes.get_lines()) == 2, "The loss curve must draw training and validation."

    labels = {text.get_text() for text in axes.get_legend().get_texts()}
    assert labels == {"Training", "Validation"}


def test_confusion_matrix_figure_shows_the_metrics_counts(hand_metrics) -> None:
    """The drawn matrix holds the same counts the Metrics reports.

    The figure takes a Metrics rather than raw arrays precisely so the picture
    and the quoted numbers cannot disagree; this proves that they do not.
    """
    figure = plot_confusion_matrix(hand_metrics, "M1 Text-CNN", "validation")
    drawn = {int(float(text.get_text())) for text in figure.axes[0].texts}

    assert drawn == {
        hand_metrics.true_negatives,
        hand_metrics.false_positives,
        hand_metrics.false_negatives,
        hand_metrics.true_positives,
    }
    assert f"{hand_metrics.threshold:.2f}" in figure.axes[0].get_title()


def test_loss_curve_rejects_unusable_history() -> None:
    """An empty history or missing columns fail rather than drawing a blank."""
    with pytest.raises(ValueError, match="empty history"):
        plot_loss_curve([], "M1 Text-CNN")

    with pytest.raises(ValueError, match="val_loss"):
        plot_loss_curve([{"epoch": 1, "train_loss": 0.5}], "M1 Text-CNN")


# --------------------------------------------------------------------------
# Reporting tables and metric persistence.
# --------------------------------------------------------------------------


def test_metrics_table_puts_one_model_per_row(hand_metrics) -> None:
    """The M0 vs M1 comparison reads one row per model, metrics as columns."""
    table = metrics_table({"M0 TF-IDF + LogReg": hand_metrics, "M1 Text-CNN": hand_metrics})

    assert list(table.index) == ["M0 TF-IDF + LogReg", "M1 Text-CNN"]
    assert table.index.name == "model"
    for column in ("accuracy", "precision", "recall", "f1", "macro_f1", "auc_roc"):
        assert column in table.columns
    assert table.loc["M1 Text-CNN", "precision"] == pytest.approx(0.6)


def test_per_corpus_table_is_indexed_by_corpus(hand_metrics) -> None:
    """The breakdown is indexed by corpus name and sorted, matching loco_report."""
    table = per_corpus_table(
        {"Nazario": hand_metrics, "CEAS_08": hand_metrics, "Enron": hand_metrics}
    )

    assert list(table.index) == ["CEAS_08", "Enron", "Nazario"]
    assert table.index.name == "corpus"


def test_empty_results_are_rejected() -> None:
    """No results means a blank table in the report; fail instead."""
    with pytest.raises(ValueError, match="at least one entry"):
        metrics_table({})


def test_save_metrics_embeds_provenance(tmp_path: Path, hand_metrics) -> None:
    """The written JSON carries the commit, seed and config hash beside the scores.

    A number in the report has to trace back to the run that produced it, which
    is what makes the result defensible rather than merely stated.
    """
    path = tmp_path / "nested" / "m1_text_cnn.json"
    save_metrics(
        hand_metrics,
        path,
        {
            "model": "m1_text_cnn",
            "split_evaluated": "test",
            "seed": 42,
            "commit_hash": "abc123",
            "config_hash": "3f78ae66c380fce1",
        },
    )

    written = json.loads(path.read_text(encoding="utf-8"))

    assert written["commit_hash"] == "abc123"
    assert written["config_hash"] == "3f78ae66c380fce1"
    assert written["seed"] == 42
    assert written["threshold"] == pytest.approx(0.5)
    assert written["metrics"]["precision"] == pytest.approx(0.6)
    assert written["confusion_matrix"]["false_positives"] == 2
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_save_metrics_refuses_a_context_that_contradicts_the_scores(
    tmp_path: Path, hand_metrics
) -> None:
    """A context threshold could disagree with the one the scores used."""
    with pytest.raises(ValueError, match="threshold"):
        save_metrics(hand_metrics, tmp_path / "x.json", {"threshold": 0.9})


def test_commit_hash_returns_a_sha_or_the_sentinel() -> None:
    """Either a real commit, or "unknown" outside a repository : never a crash."""
    value = commit_hash()

    assert value == UNKNOWN_COMMIT or (len(value) == 40 and value.isalnum())
