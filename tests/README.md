# tests/

Two folders, because two kinds of test carry different weight here.

| Folder | Contents |
|--------|----------|
| `unit/` | Ordinary unit tests for the core library: config validation, normalisation, encoding, plotting, API request handling |
| `critical/` | The guards. If one of these is wrong, every number in the report is wrong and nothing else in the test suite would notice |

## The guards

| File | What it proves |
|------|----------------|
| `test_split_integrity.py` | No content-hash group appears in more than one split; no exact duplicate text spans two splits; label proportions are preserved within tolerance; the split reproduces byte for byte from the same seed |
| `test_corpus_probe.py` | Trains a TF-IDF + Logistic Regression probe to predict an email's source corpus and records its accuracy to `artifacts/metrics/corpus_probe.json`. A **high** score is expected and is a reported finding, not a failure. The test fails only if the probe cannot be built or the result is not recorded |
| `test_metric_correctness.py` | Precision, recall, F1 and AUC computed by the evaluation layer match values worked out by hand on a small fixed confusion matrix, and the positive class is asserted to be label `1`. Catches a silently flipped positive class |
| `test_early_stopping.py` | On a synthetic run whose validation score worsens from a known epoch, training stops after exactly `patience` epochs and the returned weights are the best epoch's, not the last |

## Rules

- `pytest tests/critical` must be green before any number is quoted anywhere:
  in a notebook, in the report, in a presentation.
- A later session must not weaken or delete a guard to make a result look
- Tests use small fixtures and a fixed seed. No test downloads data or needs a
  trained model.

## Integration

Tests import from `src/phishlens/` only. They do not import notebooks, and they
do not require a running API except `tests/unit/test_api.py`, which uses the
FastAPI `TestClient` rather than a live server.

Done when: `pytest -q` is green from a clean checkout after `pip install -r
requirements.txt`, with no network access.
