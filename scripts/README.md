# scripts/

Small operational helpers. Each is a thin wrapper around logic in
`src/phishlens/`; there is no business logic in this folder.

## Files

| File | Responsibility |
|------|----------------|
| `download_data.py` | Fetch the Kaggle corpora into `data/raw/`, verify the expected files are present and report row counts. Needs a Kaggle API token in the standard location (never in the repository) |
| `build_splits.py` | Clean, de-duplicate and write the grouped train/validation/test split to `data/processed/` |
| `check_notebooks.py` | Assert every notebook in `notebooks/` has stored outputs and no stored errors, and that only `04_evaluate.ipynb` reads the test split (the LOCO notebook's pool reassembly is recognised and allowed) |
| `deploy_space.py` | Assemble a minimal folder for a Hugging Face Docker Space (code, config, served model and threshold only) and optionally upload it. Kept as an alternative host; the live demo runs on Render (`render.yaml`) |

## Integration

Depends on `src/phishlens/`. Nothing depends on this folder: removing it does
not break the library, the API or the apps.
