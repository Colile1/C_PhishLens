# src/phishlens/models/

Model definitions only. No training loop, no data loading, no metrics : those
live in `training/` and `evaluation/`. Keeping this folder to architecture alone
is what lets the architecture be read, described in report §3.2, and unit-tested
with a random tensor.

## Planned files

| File | Responsibility |
|------|----------------|
| `text_cnn.py` | `TextCNN(vocab_size, embedding_dim, filter_sizes, filters_per_size, dropout)` : the M1 architecture: embedding → parallel 1D convolutions of width 3, 4 and 5 → ReLU → max-over-time pooling → concatenate → dropout → linear → single logit. `forward(x: Tensor) -> Tensor` returns logits, not probabilities |
| `baseline.py` | `build_baseline(config) -> Pipeline` : the M0 sklearn pipeline, TF-IDF into Logistic Regression with class weighting. A factory, not a trained object |
| `registry.py` | `build_model(name: str, config) -> Module` : map `config.model.name` to a constructor, with an error naming the unknown model and listing the valid ones |
| `persistence.py` | `save_model(model, path, metadata) -> None` and `load_model(path) -> tuple[Module, dict]` : weights plus a metadata block carrying the commit hash, the seed, the vocabulary hash and the threshold. The API refuses to start if the vocabulary hash does not match the one on disk |

## Design notes for report §3.2

- The output is a single logit with `BCEWithLogitsLoss`, not a softmax over two
  classes. Numerically steadier, and it makes the decision threshold explicit
  rather than hidden inside an argmax.
- Filter widths 3, 4 and 5 are phrase detectors. Phishing lures are phrases
  ("verify your account within 24 hours"), which is the argument for a CNN over
  a recurrent model on this data.
- Max-over-time pooling makes the model indifferent to *where* in a long email
  the lure appears, which matters because it can appear anywhere.

## Integration

Imports `config` and, for tensor shapes, nothing else. Imported by `training`,
`evaluation`, `explain` and `serving`. Must not import `data` or `training` :
a model that knows how it is trained cannot be swapped.

Done when: a forward pass on a random integer tensor of shape
`(batch, max_sequence_length)` returns shape `(batch,)`, the parameter count is
printed for the report, and `registry.build_model("nope", config)` errors naming
the valid options.
