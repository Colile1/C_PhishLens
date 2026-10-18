# notebooks/

The executable record of the project. Every metric, figure and training curve
in the report is produced here and left visible, so the results can be read
without running anything.

Rules for this folder:

- **Always committed with outputs.** Never cleared, never a notebook that was
  not run. `scripts/check_notebooks.py` checks this.
- **Re-run from a clean kernel** whenever code a notebook imports has changed
  behaviour. Stale outputs are the same failure as missing ones.
- **The first cell of every notebook** records the commit hash, the seed and the
  date, so a figure can be traced to the code that made it.
- **The test split is loaded in `04_evaluate.ipynb` only**, once. No other
  notebook touches it.
- Notebooks import from `src/phishlens/`; they do not hold copies of that logic,
  so the same code is unit-tested and served by the API. The model class, the
  training loop and the attribution code are also printed with
  `inspect.getsource(...)`, so they can be read without leaving the notebook.
- Each `.ipynb` has a paired `.py` (jupytext percent format) with the same
  cells, which diffs cleanly in git.

## Files

| File | Contents |
|------|----------|
| `01_data_audit.ipynb` | Per-corpus row counts, class balance, empty and corrupted rows, duplicate rate, text-length distributions, example emails from each class |
| `02_baseline.ipynb` | M0: TF-IDF + Logistic Regression. Full metric set, ROC, precision–recall, confusion matrix. The number M1 has to beat |
| `03_train_cnn.ipynb` | M1: Text-CNN training. Per-epoch metric table, training and validation loss curves, the epoch early stopping fired on, the printed model and loop |
| `04_evaluate.ipynb` | Test-set evaluation, once. The four figures, per-corpus breakdown, M0 against M1 comparison, false-positive review |
| `05_loco.ipynb` | Leave-one-corpus-out: six folds, six scores, the gap against the headline number |
| `06_xai.ipynb` | LIME explanations for M0 and M1 on the same emails, the stability check across seeds, and the Integrated Gradients cross-check for M1 |

## Integration

Imports `src/phishlens/`. Writes figures to `artifacts/figures/` and metric
files to `artifacts/metrics/`. Reads split files from `data/processed/`, and raw
data only in notebook 01.

To re-run one: `jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb`.
`05_loco.ipynb` trains six models and takes many hours on a CPU.
