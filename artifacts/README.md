# artifacts/

Generated outputs. Split by whether the report quotes them.

| Subfolder | Contents | Committed |
|-----------|----------|-----------|
| `models/` | Trained weights: `text_cnn_best.pt`, the fitted TF-IDF vectoriser, the vocabulary | No — rebuilt from the notebooks |
| `figures/` | Saved copies of the ROC, precision–recall, confusion matrix and loss curves | No — they also live inline in the notebooks, which is what the reader reads |
| `metrics/` | Per-epoch metric CSVs, final metric JSON per model, `corpus_probe.json`, the LOCO table | **Yes** — the report quotes these numbers, so they are versioned alongside the commit that produced them |

## Rules

- A model file is only written by the training notebook, and only the
  best-epoch weights, never the last epoch.
- Metric files carry the commit hash and the seed in their contents, not only in
  their filename, so a number in the report can be traced to a run.
- Never edit a metric file by hand. If a number is wrong, re-run the notebook.

## Integration

Written by `src/phishlens/training/` and `src/phishlens/evaluation/`, read by
`src/phishlens/serving/` (the model file) and by the report.

Done when: `artifacts/metrics/` contains a metric file for M0 and M1, the corpus
probe and the LOCO runs, and every number quoted in the report appears in one of
them.
