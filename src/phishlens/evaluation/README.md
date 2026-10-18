# src/phishlens/evaluation/

Metric computation, threshold selection and the four Matplotlib figures. This
layer is **correctness-critical**: a flipped positive class or the wrong
probability column produces numbers that look entirely plausible and are wrong.

## Files

| File | Responsibility |
|------|----------------|
| `metrics.py` | `compute_metrics(labels, probabilities, threshold) -> Metrics` : accuracy, precision, recall, F1, macro-F1, AUC-ROC, average precision and the four confusion-matrix counts. The positive class is asserted, not assumed. `select_threshold(labels, probabilities, objective)` picks an operating point on the **validation split only**. Pure: arrays in, a dataclass out |
| `plots.py` | `plot_roc`, `plot_precision_recall`, `plot_confusion_matrix`, `plot_loss_curve` : each returns a Matplotlib `Figure`, so a notebook displays it inline and the caller decides whether to save |
| `reporting.py` | `metrics_table` for the M0 against M1 comparison and `per_corpus_table` for the per-corpus breakdown. Formatting only |
| `persistence.py` | `save_metrics(metrics, path, context)` : write the metric JSON with the commit hash, seed and config hash embedded, so a number in the report traces back to a run |

## Rules

- Metrics take **probabilities**, not hard labels, so AUC and the PR curve are
  computable and the threshold stays an explicit argument.
- Results are reported at 0.5 **and** at the macro-F1-maximising threshold
  selected on validation. Selecting against test would turn a reported number
  into a tuned one.
- Every figure carries axis labels, a title naming the model and the split, and
  a legend where more than one series is drawn.
- Figures are displayed inline in the notebooks and saved to
  `artifacts/figures/`.

## Integration

Imports `config` only. Used by the evaluation notebooks. Must not import
`models`, `data` or `serving`: it works on arrays, which is what makes it
testable by hand.

Verified by `tests/critical/test_metric_correctness.py`: on a fixed confusion
matrix worked out by hand, precision, recall and F1 match to six decimal
places, the positive class is label `1`, and every figure has labelled axes.
