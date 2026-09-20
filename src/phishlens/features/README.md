# src/phishlens/features/

Turns cleaned text into what each model eats: a sparse TF-IDF matrix for M0, and
padded integer sequences for M1.

The single rule this module must never break: **everything is fitted on the
training split only.** A vectoriser or vocabulary fitted on the full frame leaks
the validation and test distributions into the model and is invisible in the
results.

## Planned files

| File | Responsibility |
|------|----------------|
| `normalise.py` | `normalise_text(text: str, config) -> str` : lowercase, replace URLs with `<url>`, email addresses with `<email>`, long digit runs with `<num>`, collapse whitespace. Pure and idempotent: `normalise(normalise(x)) == normalise(x)` |
| `tokenise.py` | `tokenise(text: str) -> list[str]` : simple whitespace and punctuation tokeniser. Deliberately plain, so the tokens LIME reports are the tokens the model saw |
| `vectoriser.py` | `fit_tfidf(train_texts, config) -> TfidfVectorizer` and `transform(vectoriser, texts)` : the M0 feature path. Fit takes training texts only; the signature makes misuse awkward |
| `vocabulary.py` | `build_vocabulary(train_texts, config) -> Vocabulary` : token to index map with `<pad>` at 0 and `<unk>` at 1, capped at `max_size`, filtered by `min_frequency`. `save` and `load` so the API uses the exact vocabulary the model trained on |
| `encoder.py` | `encode(texts, vocabulary, max_length) -> numpy.ndarray` : integer sequences, padded and truncated to `max_sequence_length`. Round-trips a known sentence; unknown tokens map to `<unk>` |

## Integration

Imports `config` and `data`. Imported by `models`, `training`, `explain` and
`serving`. The fitted vectoriser and the vocabulary are saved to
`artifacts/models/` so the API reconstructs the exact same feature space : a
mismatch here is a class of bug that produces confident nonsense at serving time
while every offline test still passes.

Done when: the normaliser is proved idempotent by test; the encoder round-trips
a known sentence; a token absent from the vocabulary maps to `<unk>`; and a test
asserts the vectoriser is fitted on training texts only.
