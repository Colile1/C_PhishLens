# src/phishlens/models/

Model definitions only. No training loop, no data loading, no metrics: those
live in `training/` and `evaluation/`. Keeping this folder to architecture alone
is what lets the architecture be read on its own, described in the report, and
unit-tested with a random tensor.

## Files

| File | Responsibility |
|------|----------------|
| `text_cnn.py` | `TextCNN(vocab_size, embedding_dim, filter_sizes, filters_per_size, dropout)` : embedding → parallel 1D convolutions of width 3, 4 and 5 → ReLU → max-over-time pooling → concatenate → dropout → linear → one logit. `build_text_cnn(vocab_size, config)` builds it from config; `count_parameters(model)` reports its size |
| `baseline.py` | `build_baseline(config) -> Pipeline` : the M0 sklearn pipeline, TF-IDF into Logistic Regression, class-weighted when `training.class_weighting` is set. A factory, not a trained object |

## Design notes

- The output is a single logit trained with `BCEWithLogitsLoss`, not a softmax
  over two classes: numerically steadier, and it keeps the decision threshold
  explicit rather than hidden inside an argmax.
- Filter widths 3, 4 and 5 act as phrase detectors. Phishing lures are phrases
  ("verify your account within 24 hours"), which is the argument for a CNN on
  this data.
- Max-over-time pooling makes the model indifferent to *where* in a long email
  the lure appears.

## Integration

Imports `config` and the vocabulary's padding index. Imported by `training`,
`explain` and `serving`. Must not import `data` or `training`: a model that
knows how it is trained cannot be swapped.
