# artifacts/

Generated outputs, split by whether the report quotes them.

| Subfolder | Contents | Committed |
|-----------|----------|-----------|
| `models/` | Trained weights: `text_cnn_best.pt`, the vocabulary, and the LOCO fold checkpoints under `loco/` | **Partly**: the served M1 checkpoint `text_cnn_best.pt` and `vocabulary.json` are committed so a clean clone can run the API and the apps without retraining; the LOCO fold checkpoints are rebuilt by `notebooks/05_loco.ipynb` |
| `figures/` | Saved copies of the ROC, precision–recall, confusion matrix, loss and explanation figures | No: they are also inline in the notebooks |
| `metrics/` | Per-epoch metric CSVs, the final metric JSON per model, `corpus_probe.json`, the LOCO results, the XAI stability and LIME/IG agreement files | **Yes**: the report quotes these numbers, so they are versioned with the commit that produced them |

## Rules

- A model file is written only by the training notebook, and only with the
  best-epoch weights, never the last epoch.
- Metric files carry the commit hash and the seed in their contents, so a number
  in the report can be traced to a run.
- Never edit a metric file by hand. If a number is wrong, re-run the notebook.
- If M1 is ever retrained, commit `text_cnn_best.pt`, `vocabulary.json`,
  `m1_text_cnn.json` and the notebooks together.

## Integration

Written by `src/phishlens/training/` and `src/phishlens/evaluation/` through the
notebooks; read by `src/phishlens/serving/` (model, vocabulary, threshold) and
by the report.
