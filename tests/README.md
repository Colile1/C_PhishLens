# tests/

Two folders, because two kinds of test carry different weight here.

| Folder | Contents |
|--------|----------|
| `unit/` | Ordinary unit tests: config validation, data layer, features, the API, the browser UI, the desktop client, the notebook audit |
| `critical/` | The guards. If one of these is wrong, every reported number is wrong and nothing else in the suite would notice |

## The guards

| File | What it proves |
|------|----------------|
| `test_split_integrity.py` | No content-hash group appears in more than one split; no exact duplicate text spans two splits; label proportions hold within tolerance; the split reproduces from the same seed. Skips cleanly when `data/processed/` is absent |
| `test_corpus_probe.py` | Trains a TF-IDF + Logistic Regression probe to predict an email's source corpus and records its accuracy to `artifacts/metrics/corpus_probe.json`. A **high** score is expected and is a reported finding, not a failure |
| `test_metric_correctness.py` | Precision, recall, F1 and AUC match values worked out by hand on a fixed confusion matrix, and the positive class is label `1` |
| `test_early_stopping.py` | Training stops after exactly `patience` epochs without improvement, and the returned weights are the best epoch's, not the last |
| `test_attribution_alignment.py` | LIME weights land on the right tokens, at the right character offsets, in order |
| `test_ig_alignment.py` | Integrated Gradients scores land only on the tokens that moved the logit, with the project's sign, and sum to the logit gap |

## Rules

- `pytest tests/critical` must be green before any number is quoted anywhere.
- Never weaken or delete a guard to make a result look better. If a guard is
  wrong, fix the guard and say why in the commit.
- Tests use small fixtures and a fixed seed. No test downloads data. Tests that
  need the split or the committed checkpoint skip when those are absent.

## Running

```bash
pytest -q                   # everything
pytest tests/critical -q    # the guards only
```

The corpus probe writes `artifacts/metrics/corpus_probe.json` each run; its
fourth decimal can differ across Python builds. The committed file is the one
the report quotes.
