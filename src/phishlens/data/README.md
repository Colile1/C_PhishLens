# src/phishlens/data/

Turns six raw corpus CSVs into three split files that can be trusted. This
layer is **correctness-critical**: a subtle error here inflates every reported
metric and nothing downstream would notice.

## Files

| File | Responsibility |
|------|----------------|
| `loader.py` | `load_corpora(config) -> DataFrame` : read each CSV in `config.data.corpora`, add a `corpus` column naming the source, harmonise columns, concatenate. Raises an error listing any expected file that is missing |
| `cleaning.py` | `clean(frame, config)` : strip HTML, collapse whitespace, join subject and body with the separator, drop rows shorter than `min_characters`, with a null label, or carrying mbox parser residue. Returns the frame plus a count of what was dropped and why |
| `hashing.py` | `content_hash(text) -> str` : normalise aggressively (lowercase, strip punctuation and digits, collapse whitespace) then hash. Two emails differing only in a name, a date or an amount hash the same, because those are exactly the near-duplicates that leak |
| `dedup.py` | Attach a `content_hash` group id and report the duplicate rate per corpus. Rows are not deleted: grouping is what prevents leakage, deletion would only discard data |
| `splitter.py` | Grouped, stratified 70/15/15 split. Groups are assigned whole. Writes the split files plus a manifest recording the seed, the config hash and the realised proportions; `load_split` reads them back |
| `loco.py` | Leave-one-corpus-out index: six train/test pairs, one per held-out corpus, for `notebooks/05_loco.ipynb` |

## The two rules this layer enforces

1. A content-hash group is assigned entirely to one split. Never partially.
2. The split is drawn once, written to disk, and reused. Nothing redraws it at
   training time, so every model is compared on identical data.

## Integration

Imports `config` only. Imported by `features`, by the notebooks and scripts, and
by the guards in `tests/critical/`. Must not import `models`, `training` or
`serving`.

Verified by `tests/critical/test_split_integrity.py` (no group spans two
splits, no exact duplicate text spans two splits, label proportions hold within
one percentage point, the split reproduces from the same seed) and
`tests/unit/test_data.py`.
