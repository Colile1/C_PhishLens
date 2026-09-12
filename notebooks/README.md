# notebooks/

The executable record of the project. The brief requires model code in a
Jupyter notebook with all figures and evaluation metrics viewable directly
inside it, and attaches a −10 deduction if they are not. These notebooks are
therefore the primary marked artefact, not a scratchpad.

Rules for this folder:

- **Always commit with outputs.** Never clear them, never commit a notebook that
  was not run.
- **Re-run from a clean kernel** whenever anything in `src/` that a notebook
  imports has changed. Stale outputs are the same failure as missing ones.
- **First cell of every notebook** records the commit hash, the config file used,
  the seed and the date, so a figure can be traced to the code that made it.
- **The test split is loaded in `04_evaluate.ipynb` only**, once. No other
  notebook touches it. This is checked.
- Notebooks import from `src/phishlens/`; they do not hold copies of that logic.
  The model class and the training loop are additionally printed with
  `inspect.getsource(...)` so the reader reads them without leaving the notebook.

## Planned files

| File | Responsibility |
|------|----------------|
| `01_data_audit.ipynb` | Per-corpus row counts, class balance, empty and corrupted rows, duplicate rate, text-length distributions, example emails from each class. Produces the evidence that decides the question |
| `02_baseline.ipynb` | M0: TF-IDF + Logistic Regression. Full metric set, ROC, precision–recall, confusion matrix. The number M1 has to beat |
| `03_train_cnn.ipynb` | M1: Text-CNN training. Per-epoch metric table, training and validation loss curves, the epoch early stopping fired on, `inspect.getsource` of the model and the loop |
| `04_evaluate.ipynb` | Test-set evaluation, once. The four required graphs, per-corpus breakdown, M0 vs M1 comparison, false-positive review |
| `05_loco.ipynb` | Leave-one-corpus-out: six runs, six F1 scores, the gap against the headline number |
| `06_xai.ipynb` | LIME explanations for M0 and M1 on the same emails, the stability check across two seeds, and the explanations behind the false positives from notebook 04 |

## Integration

Imports `src/phishlens/`. Writes figures to `artifacts/figures/` and metric
files to `artifacts/metrics/`. Reads split files from `data/processed/`, never
raw data directly except in notebook 01.

Done when: all six run top to bottom from a clean kernel, are committed with
outputs, and a reader can read every required metric and figure without
executing a single cell.
