# C_PhishLens

**Project:** ITRI625 Machine Learning project — North-West University, Semester 2, 2026
**Author:** Colile Sibanda
**Repository:** https://github.com/Colile1/C_PhishLens
**Deadline:** _TBD in the brief — confirm and record here_

A deep learning classifier that decides whether an email is phishing or
legitimate from its subject and body text alone, and returns the tokens that
drove the decision. It is served by a FastAPI endpoint and used through a
Tkinter desktop application, so the model is something an analyst can actually
open and paste a message into.

The project's central claim is not the headline F1. Pooled phishing corpora
hand you a high score cheaply, because a model can learn which corpus an email
came from instead of learning phishing. C_PhishLens measures that risk with a
source-corpus probe and a leave-one-corpus-out evaluation, and reports both
next to the headline number.

## What it does

- Classifies an email as phishing or legitimate from text alone — no headers, no URL features, no sandbox.
- Logs accuracy, precision, recall, F1, macro-F1, AUC-ROC and loss after every training epoch.
- Plots ROC, precision–recall, confusion matrix and loss curves with Matplotlib, visible inside the notebooks.
- Stops training on a patience value implemented in the main loop and restores the best weights before the test set is touched.
- Explains a single prediction at token level with LIME, so the verdict is checkable.
- Serves the model over FastAPI and drives a Tkinter desktop client against it.

## Layout

| Path | Purpose |
|------|---------|
| `run_api.py` | Thin entry point — starts the FastAPI service (built at step 13) |
| `run_app.py` | Thin entry point — starts the Tkinter client (built at step 14) |
| `configs/` | `default.yaml`: seed, paths, split ratios, model and training hyperparameters |
| `notebooks/` | The executable record: audit, baseline, training, evaluation, LOCO, XAI |
| `src/phishlens/` | Core library — config, data, features, models, training, evaluation, explain, serving |
| `src/desktop/` | Tkinter presentation layer |
| `tests/unit/` | Unit tests for the core library |
| `tests/critical/` | The guards: split integrity, corpus probe, metric correctness |
| `data/` | `raw/` downloaded corpora, `interim/` cleaned, `processed/` split files — none committed |
| `artifacts/` | Trained models, per-epoch metric CSVs, saved figures — none committed |
| `scripts/` | Small operational helpers (download, smoke test) |

## Quick start (once built)

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows;  source .venv/bin/activate on Linux
pip install -r requirements.txt

pytest -q                          # all tests
pytest tests/critical -q           # the guards

jupyter lab notebooks/             # read the results without running anything

python run_api.py                  # FastAPI on http://127.0.0.1:8000
python run_app.py                  # Tkinter client, in a second terminal
```

## Reproducibility contract

Every number in the report is reproducible from three things: the committed
`configs/default.yaml`, the pinned `requirements.txt`, and the commit hash
recorded in `/metadata` and in each notebook's header cell.

One seed drives every source of randomness — `random`, `numpy`, `torch`, the
splitter and LIME. The split is written to disk, so the same split is reused by
every model rather than redrawn. The test set is loaded in exactly one
notebook, once, at the end.

## Quality contract

- `pytest tests/critical` must be green before any result is quoted anywhere.
- No content-hash group appears in more than one split. The test enforces it.
- Vectorisers and vocabularies are fitted on training data only.
- Reported test metrics come from the restored best-epoch weights, never the last epoch.
- Notebooks are committed with outputs, re-run from a clean kernel whenever `src/` changes.

## Status

repository pushed to GitHub. No application code written yet. Start at build


from the repository root.
