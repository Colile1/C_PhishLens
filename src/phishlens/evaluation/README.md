# src/phishlens/evaluation/

Metric computation and the four Matplotlib graphs the brief requires. This is
**correctness-critical**: a flipped positive class or the wrong
probability column produces numbers that look entirely plausible and are wrong,
and every one of them would end up in the report.

## Planned files

| File | Responsibility |
|------|----------------|
| `metrics.py` | `compute_metrics(y_true, y_prob, threshold) -> Metrics` — accuracy, precision, recall, F1, macro-F1, AUC-ROC, and the confusion matrix counts. The positive class is asserted, not assumed. Pure: takes arrays, returns a dataclass, touches no files |
| `plots.py` | `plot_roc(...)`, `plot_precision_recall(...)`, `plot_confusion_matrix(...)`, `plot_loss_curve(...)` — each returns a Matplotlib `Figure` so a notebook can display it and the caller decides whether to save. Axis labels, titles and legends are mandatory, not optional |
| `reporting.py` | `metrics_table(results: dict[str, Metrics]) -> DataFrame` for the M0 vs M1 comparison, and `per_corpus_table(...)` for the breakdown. Formatting only |
| `persistence.py` | `save_metrics(metrics, path, context) -> None` — write the JSON with the commit hash, seed and config hash embedded, so a number in the report traces back to a run |

## Rules

- Metrics take **probabilities**, not hard labels, so AUC and the PR curve are
  computable and the threshold stays an explicit argument.
- The threshold comes from config, defaults to 0.5, and is reported next to
  every metric that depends on it. See open question the question.
- Every figure carries axis labels, a title naming the model and the split, and
  a legend where more than one series is drawn. A reader should not have to ask
  what a figure shows.
- Figures are returned, then both displayed inline in the notebook and saved to
  `artifacts/figures/`. Inline is what matters.

## Integration

Imports `config` only. Imported by `training` (per-epoch metrics) and by the
evaluation notebooks. Must not import `models`, `data` or `serving` — it works
on arrays, which is what makes it testable by hand.

Done when: on a fixed 2×2 confusion matrix worked out by hand, precision, recall
and F1 match to six decimal places; a test asserts the positive class is label
`1`; and all four figures render with labelled axes. Test lives in
`tests/critical/test_metric_correctness.py`.
