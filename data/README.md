# data/

All data, at three stages. **Nothing in this folder is committed** except the
`.gitkeep` files — the corpora are public and reproducible with
`scripts/download_data.py`, and committing them would bloat the repository and
muddy the licensing position.

| Subfolder | Contents | Produced by |
|-----------|----------|-------------|
| `raw/` | The six downloaded corpus CSVs, exactly as obtained. Never edited in place | `scripts/download_data.py` |
| `interim/` | Cleaned and normalised rows with the content hash attached, before splitting | `src/phishlens/data/` |
| `processed/` | The split files: train, validation and test, plus the leave-one-corpus-out index. These are what every model reads | `src/phishlens/data/splitter.py` |

## Rules

- `raw/` is read-only once downloaded. Every transformation writes to `interim/`.
- The split is drawn **once**, written to `processed/`, and reused by every model.
  Nothing redraws it at training time.
- Split files record the seed and the config hash they came from, so a mismatch
  is detectable rather than silent.
- The test split is loaded by exactly one notebook, at the end.

## Licensing

The aggregate download carries no single licence; each source corpus has its own
terms. Record the access link and the per-corpus conditions in
`docs/dataset-card.md`, before any modelling. The data derives
from public email archives. No sample is ever executed, no address in it is ever
contacted, and no individual email is reproduced in the report without removing
personal detail.

Done when: `data/processed/` holds split files that reproduce byte for byte from
the same seed, and `tests/critical/test_split_integrity.py` passes against them.
