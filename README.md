# C_PhishLens

A deep learning classifier that decides whether an email is phishing or
legitimate from its subject and body text alone, and shows the words that drove
the decision, so an analyst can check the verdict instead of trusting it.

**Live demo:** https://phishlens-vdc9.onrender.com (free hosting: the first visit
after 15 idle minutes takes about a minute to wake)
**Author:** Colile Sibanda · ITRI625 Machine Learning project, North-West University, 2026
**Repository:** https://github.com/Colile1/C_PhishLens

The headline score is not the point of the project. Pooled phishing corpora give
a high score cheaply, because a model can learn *which archive* an email came
from instead of learning phishing. C_PhishLens measures that risk with a
source-corpus probe and a leave-one-corpus-out evaluation, and reports both next
to the headline number.

## Results at a glance

| | M0: TF-IDF + Logistic Regression | M1: Text-CNN (served) |
|---|---|---|
| Test macro-F1 | 0.9856 | **0.9871** |
| Test recall (phishing) | 0.9866 | **0.9925** |
| Test AUC-ROC | 0.9988 | **0.9991** |

- **Corpus probe:** a linear model recovers an email's source archive from text
  alone at 0.9445 accuracy, so part of any random-split score is archive
  recognition.
- **Leave-one-corpus-out:** with one whole archive held out, macro-F1 falls to
  between 0.7706 and 0.9357 on the archives that contain both classes, and
  recall on unseen real-world phishing (Nazario) is 0.5202.
- **Explanations:** LIME token attributions, checked for stability across seeds
  and cross-checked with Integrated Gradients.

The full analysis is in `docs/report/`. Every number above is in
`artifacts/metrics/` and visible in the notebooks.

## What it does

- Classifies an email as phishing or legitimate from text alone: no headers, no
  URL features, no sandbox. Headers carry archive-specific residue, which is the
  shortcut the model is kept away from.
- Logs accuracy, precision, recall, F1, macro-F1, AUC-ROC and loss after every
  training epoch.
- Plots ROC, precision–recall, confusion-matrix and loss curves with Matplotlib,
  visible inside the notebooks.
- Stops training on a patience value implemented in the main training loop and
  restores the best weights before the test set is touched.
- Explains each prediction at token level (LIME), with an Integrated Gradients
  cross-check in the XAI notebook.
- Serves the model over FastAPI, with a Tkinter desktop client and a browser UI, and
  runs inside a notebook page too (`notebooks/07_analyst_ui.ipynb`), readable on GitHub
  and usable in Jupyter with no server.
- Runs publicly on Render, built from this repository by `render.yaml` and
  `deploy/render/Dockerfile`, with rate limits and security headers.

## Layout

| Path | Purpose |
|------|---------|
| `run_api.py` | Thin entry point: starts the FastAPI service |
| `run_app.py` | Thin entry point: starts the Tkinter client |
| `configs/` | `default.yaml`: seed, paths, split ratios, model, training, serving settings |
| `notebooks/` | The executable record: audit, baseline, training, evaluation, LOCO, XAI, and the analyst UI page |
| `src/phishlens/` | Core library: config, data, features, models, training, evaluation, explain, serving |
| `src/desktop/` | Tkinter client |
| `src/web/` | Browser UI, served by the API at `/ui/` |
| `src/notebook_ui/` | Notebook client used by `notebooks/07_analyst_ui.ipynb` |
| `tests/unit/`, `tests/critical/` | Unit tests, and the guards every reported number depends on |
| `data/` | `raw/`, `interim/`, `processed/`: none committed, rebuilt by `scripts/` |
| `artifacts/` | Served model and vocabulary, metric files (committed); figures and LOCO checkpoints (rebuilt) |
| `docs/` | Dataset card, references, the theory report |
| `deploy/` | Docker images for hosting (Render; a Hugging Face Space variant) |
| `scripts/` | Data download, split build, notebook audit, Space packaging |

Every folder has a README describing its files.

## Quick start

Reading the results needs nothing but the repository: every notebook is
committed with its outputs, and every metric the report quotes is in
`artifacts/metrics/`.

```bash
python -m venv .venv
.venv\Scripts\activate             # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt

pytest -q                          # all tests
pytest tests/critical -q           # the guards

jupyter lab notebooks/             # read the results without running anything
```

On a fresh clone without `data/`, the split-integrity and corpus-probe guards
skip with a message naming the missing file. `python scripts/check_notebooks.py`
confirms every notebook has visible outputs and that only `04_evaluate.ipynb`
evaluates the test split.

The trained model is committed (`artifacts/models/text_cnn_best.pt` and
`vocabulary.json`), so the API and the apps run straight from a clone:

```bash
python run_api.py                  # FastAPI on http://127.0.0.1:8000, browser UI at /ui/
python run_app.py                  # Tkinter client, in a second terminal
```

Paste an email (or open a `.eml` / `.txt` file) and press **Analyse** or
Ctrl+Enter. The verdict comes with its probability and decision threshold, the
words that pushed towards each class are highlighted, and the strongest signals
are listed. Screenshots: `docs/report/desktop-analysed-phishing.png` and
`docs/report/desktop-api-down.png`.

Or work entirely in a notebook: open `notebooks/07_analyst_ui.ipynb` in Jupyter,
run the setup cell, then call `analyse(text)` or use the interactive panel. The
model runs in the notebook itself, so no server is needed.

To rebuild the data and retrain from scratch:

```bash
python scripts/download_data.py    # Kaggle corpora into data/raw/ (needs a Kaggle API token)
python scripts/build_splits.py     # cleaning, dedup, grouped train/validation/test split
jupyter nbconvert --to notebook --execute --inplace notebooks/03_train_cnn.ipynb
```

## Reproducibility

Every reported number is reproducible from the committed `configs/default.yaml`,
the pinned `requirements.txt`, and the commit hash recorded in `/metadata` and
in each notebook's header cell.

One seed drives every source of randomness: `random`, `numpy`, `torch`, the
splitter and LIME. The split is written to disk once and reused by every model.
De-duplication and splitting work on content-hash groups, never single rows, so
near-identical emails cannot leak between train and test. The test set is
loaded in exactly one notebook, once, at the end; every threshold and model
choice is made on validation.

## Quality checks

- `pytest tests/critical` is green before any result is quoted.
- No content-hash group appears in more than one split; a test enforces it.
- Vectorisers and vocabularies are fitted on training data only.
- Reported test metrics come from the restored best-epoch weights.
- Notebooks are committed with outputs. Model code lives in `src/phishlens/`, so
  it is unit-tested and shared with the API, and is printed into the notebooks
  with `inspect.getsource`: the `TextCNN` class and the `train` loop in
  `03_train_cnn.ipynb`, the attribution code in `06_xai.ipynb`.

## Security

- The public API accepts email text only, stores nothing, and caps input at
  50 000 characters.
- Scoring endpoints are rate-limited per client (429 with `Retry-After`), and
  the container runs as a non-root user.
- Every response carries security headers; the browser UI has a content security
  policy that allows no external source.
- No credentials live in the repository: the Kaggle token stays in the user's
  home directory, and `.env` files are ignored.
- Do not paste real personal email into the public demo.
