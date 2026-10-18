# src/phishlens/features/

Turns cleaned text into what each model takes: a sparse TF-IDF matrix for M0,
and padded integer sequences for M1.

The rule this layer never breaks: **everything is fitted on the training split
only.** A vectoriser or vocabulary fitted on the full frame leaks the
validation and test distributions into the model, invisibly.

## Files

| File | Responsibility |
|------|----------------|
| `normalise.py` | `normalise_text(text, config) -> str` : lowercase, replace URLs with `<url>`, email addresses with `<email>`, digit runs with `<num>`, collapse whitespace. Pure and idempotent: `normalise(normalise(x)) == normalise(x)` |
| `tokenise.py` | `tokenise(text) -> list[str]` : a plain pattern tokeniser that keeps placeholders whole, so the tokens LIME reports are exactly the tokens the encoder saw |
| `vectoriser.py` | `fit_tfidf(train_texts, config)` and `transform(vectoriser, texts, config)` : the M0 feature path. Fitting takes training texts only |
| `vocabulary.py` | `build_vocabulary(train_texts, config) -> Vocabulary` : token to index map with `<pad>` at 0 and `<unk>` at 1, capped at `max_size`, filtered by `min_frequency`. `save` and `load` so the API uses the exact vocabulary the model trained on |
| `encoder.py` | `encode(texts, vocabulary, config) -> ndarray` : integer sequences, padded and truncated to `max_sequence_length`; unknown tokens map to `<unk>` |

## Integration

Imports `config`. Imported by `training`, `explain` and `serving`. The
vocabulary is saved to `artifacts/models/` so the API reconstructs exactly the
same feature space: a mismatch here produces confident nonsense at serving time
while every offline test still passes.

Verified by `tests/unit/test_features.py`: the normaliser is idempotent, the
encoder round-trips a known sentence, an unseen token maps to `<unk>`, and the
vectoriser is fitted on training texts only.
