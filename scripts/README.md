# scripts/

Small operational helpers. Anything here is a convenience wrapper around logic
that lives in `src/phishlens/` : no business logic in this folder.

## Files

| File | Responsibility |
|------|----------------|
| `download_data.py` | `main() -> None` : fetch the Kaggle corpora into `data/raw/`, verify the expected files are present and report row counts. Fails with a named missing file rather than a partial download |
| `build_splits.py` | `main() -> None` : clean, dedup and write the grouped train/validation/test split to `data/processed/` |
| `smoke_test.py` | **Not built** (optional). `main() -> None` : start the API, post one known phishing email and one known legitimate email to `/predict` and `/explain`, assert the verdicts, print `Description: value` lines. Run this before any live demo |
| `check_notebooks.py` | `main() -> None` : assert every notebook in `notebooks/` has stored outputs and no stored errors, and that only `04_evaluate.ipynb` reads the test split (the LOCO notebook's pool reassembly is recognised and allowed). Part of the step 16 review |

## Integration

Depends on `src/phishlens/`. Nothing depends on this folder : removing it must
not break the library, the API or the app.

Done when: `python scripts/smoke_test.py` passes against a running API, and
`python scripts/check_notebooks.py` reports no notebook with cleared outputs.
