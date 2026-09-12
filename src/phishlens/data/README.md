# src/phishlens/data/

Turns six raw corpus CSVs into three split files that can be trusted. This is
**correctness-critical**: a subtle error here inflates
every metric in the report and nothing downstream would notice.

## Planned files

| File | Responsibility |
|------|----------------|
| `loader.py` | `load_corpora(config: Config) -> pandas.DataFrame` — read each CSV in `config.data.corpora`, add a `corpus` column naming the source, harmonise column names across corpora, concatenate. Raises a named error listing any expected file that is missing |
| `cleaning.py` | `clean(frame, config) -> pandas.DataFrame` — strip HTML, collapse whitespace, join subject and body with the separator, drop rows shorter than `min_characters` or with a null label. Returns the frame plus a counts report so the audit notebook can show what was dropped |
| `hashing.py` | `content_hash(text: str) -> str` — normalise aggressively (lowercase, strip punctuation and digits, collapse whitespace) then hash. Two emails differing only in a name, a date or an amount must hash the same, because those are exactly the near-duplicates that leak |
| `dedup.py` | `assign_groups(frame) -> pandas.DataFrame` — attach a `content_hash` group id and report the duplicate rate per corpus. Does not delete rows; grouping is what prevents leakage, deletion would only discard data |
| `splitter.py` | `make_split(frame, config) -> SplitIndex` — grouped, stratified 70/15/15. Groups are assigned whole. Writes split files plus a manifest recording the seed, the config hash and the realised proportions. `load_split(config) -> SplitIndex` reads them back |
| `loco.py` | `make_loco_index(frame) -> dict[str, SplitIndex]` — six train/test pairs, one per held-out corpus, |

## The two rules this module exists to enforce

1. A content-hash group is assigned entirely to one split. Never partially.
2. The split is drawn once, written to disk, and reused. Nothing redraws it at
   training time, so every model is compared on identical data.

## Integration

Imports `config` only. Imported by `features`, by every notebook, and by the
guards in `tests/critical/`. Must not import `models`, `training` or `serving`.

Done when: `tests/critical/test_split_integrity.py` passes — no group spans two
splits, no exact duplicate text spans two splits, label proportions hold within
one percentage point, and the split reproduces byte for byte from the same seed.
