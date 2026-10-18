# data/

All data, at three stages. **Nothing in this folder is committed** except the
`.gitkeep` files: the corpora are public and reproducible with
`scripts/download_data.py`, and committing them would bloat the repository and
muddy the licensing position.

| Subfolder | Contents | Produced by |
|-----------|----------|-------------|
| `raw/` | The six downloaded corpus CSVs, exactly as obtained. Never edited in place | `scripts/download_data.py` |
| `interim/` | Cleaned rows with the content hash attached, before splitting | `src/phishlens/data/` |
| `processed/` | The train, validation and test split files and their manifest. These are what every model reads | `scripts/build_splits.py` |

## Rules

- `raw/` is read-only once downloaded. Every transformation writes elsewhere.
- The split is drawn **once**, written to `processed/`, and reused by every
  model. Nothing redraws it at training time.
- The split manifest records the seed and the config hash it came from, so a
  mismatch is detectable rather than silent.
- The test split is loaded by exactly one notebook, at the end.

## Licensing and handling

The aggregate download carries no single licence; each source corpus has its own
terms, recorded in `docs/dataset-card.md`. The data derives from public email
archives. No sample is ever executed, no address in it is ever contacted, and no
individual email is reproduced in the report without removing personal detail.
