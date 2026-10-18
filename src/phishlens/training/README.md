# src/phishlens/training/

The training loop, the patience value, and the per-epoch metric log. This
layer is **correctness-critical**: a wrong best-weight restore would mean the
reported test numbers belong to a model that was never the one reported.

## Files

| File | Responsibility |
|------|----------------|
| `loop.py` | `train(...) -> TrainingResult` : the main loop. Explicit, readable, and printed into `notebooks/03_train_cnn.ipynb` with `inspect.getsource`. Holds the patience counter, the best-score tracking and the restore, and writes the per-epoch CSV |
| `early_stopping.py` | `EarlyStopping` : a pure state machine over a validation score, with `min_delta`. No tensors, so it is unit-testable on a list of floats |
| `checkpoint.py` | `save_checkpoint(model, path)` and `restore_best(model, path)` : write the best epoch's weights and load them back before evaluation. Errors if the file is missing rather than continuing with the last epoch |
| `batching.py` | `make_dataset` and `make_loader` : tensor datasets and loaders, with the generator seeded from config so batch order is reproducible |
| `epoch_metrics.py` | `compute_epoch_metrics` : the per-epoch metric row written to the training CSV |
| `weighting.py` | `positive_class_weight(labels)` : positive-class weight from the training split only |

## The behaviour that has to be exactly right

1. Patience watches `config.training.early_stopping.metric`: validation
   macro-F1, not loss. Loss and F1 disagree on imbalanced data, and the report
   is written about F1.
2. An improvement smaller than `min_delta` does not reset the counter.
3. When the loop ends, for any reason, the model carries the **best** epoch's
   weights. The test set never sees the last epoch unless it was the best.
4. The per-epoch CSV has one row per epoch with train and validation accuracy,
   precision, recall, F1, macro-F1, AUC-ROC and loss, rewritten after every
   epoch so a crashed run keeps what it completed.

## Integration

Imports `config`, `features` and `models`. Used by the training and LOCO
notebooks. Must not import `serving` or `explain`.

Verified by `tests/critical/test_early_stopping.py`: a synthetic run whose
validation score improves and then worsens stops after exactly `patience`
further epochs, and the returned weights are the best epoch's, not the last.
