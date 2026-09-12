# src/phishlens/

The core library. Everything the notebooks, the API and the tests import.

## Dependency direction

Strictly one way. Nothing to the left may import anything to its right.

```
config  →  data  →  features  →  models  →  training  →  evaluation  →  explain
                                                                          ↓
                                                                       serving
```

- `config`, `data`, `features`, `models`, `training`, `evaluation` and `explain`
  are the **core**: no HTTP, no Tkinter, no notebook imports. They must be
  testable offline with a hand-built config.
- `serving` is the **integration layer**: it imports the core and exposes it over
  HTTP. Nothing in the core imports `serving`.
- `src/desktop/` sits outside this package entirely and reaches the model only
  through the API.

There are no cycles. If a new import would create one, the shared piece belongs
in `config` or in a new small module, not in a back-reference.

## Modules

| Folder | Responsibility |
|--------|----------------|
| `config/` | Load and validate `configs/*.yaml`; seed every source of randomness from one value |
| `data/` | Load the corpora, normalise, hash for de-duplication, draw the grouped stratified split, build the leave-one-corpus-out index |
| `features/` | Text normalisation with placeholders, TF-IDF vectoriser for M0, vocabulary and integer encoder for M1 |
| `models/` | Model definitions only: the Text-CNN, and the baseline pipeline factory. No training code |
| `training/` | The training loop, patience-based early stopping, checkpointing, per-epoch metric logging |
| `evaluation/` | Metric computation and the four required Matplotlib plots |
| `explain/` | LIME wrapper producing per-token attributions, plus the stability check |
| `serving/` | FastAPI application, request and response schemas, model loading at startup |

Each has its own README with its planned files and function signatures. Read the
one for the folder you are about to touch before touching it.

## Conventions

- One responsibility per file, roughly 120 lines as a soft limit.
- Every file opens with a header: file name, one-line purpose, `Author: Colile Sibanda`.
- Every function has a docstring stating purpose, inputs and output.
- Pure functions wherever the work allows. Anything that reads a file, prints or
  plots says so in its name and lives in the layer that is allowed to do it.
- Randomness is injected, never taken from module-level global state, so tests
  are deterministic.
- Errors name the field, the file or the expected value. No bare `except`.

Done when: `pytest tests/unit tests/critical` is green and no module in the core
imports `serving`, `desktop`, or a notebook.
