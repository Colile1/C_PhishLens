# src/phishlens/training/

The training loop, the patience value, and the per-epoch metric log. This is
**correctness-critical** for two reasons: 5 points ride directly
on the early stopping, and a wrong best-weight restore means the test numbers in
the report belong to a model that was never reported.

## Planned files

| File | Responsibility |
|------|----------------|
| `loop.py` | `train(model, train_loader, val_loader, config) -> TrainingResult` : the main loop. Explicit, readable, and printed into the notebook with `inspect.getsource`. Holds the patience counter, the best-score tracking and the restore. No plotting, no file writing beyond the checkpoint |
| `early_stopping.py` | `EarlyStopping(metric, mode, patience, min_delta)` with `update(score: float) -> bool` returning whether to stop, and `best_epoch` / `best_score` properties. Pure state machine, no tensors, so it is unit-testable on a list of floats |
| `checkpoint.py` | `save_checkpoint(model, path)` and `restore_best(model, path) -> None` : write the best epoch's weights and load them back before evaluation. Errors if the file is missing rather than silently continuing with the last epoch |
| `batching.py` | `make_loaders(split, config) -> tuple[DataLoader, DataLoader, DataLoader]` : datasets and loaders, with the generator seeded from config so batch order is reproducible |
| `weighting.py` | `class_weights(labels) -> Tensor` : positive class weight from the training split only |

## The behaviour that has to be exactly right

1. Patience watches `config.training.early_stopping.metric` : validation
   macro-F1, not loss. Loss and F1 disagree on imbalanced data, and the report
   is written about F1.
2. `min_delta` means an improvement smaller than it does not reset the counter.
3. When the loop ends, for any reason, the model carries the **best** epoch's
   weights. The test set never sees the last epoch unless the last epoch was the
   best.
4. The per-epoch CSV has one row per epoch with train and validation accuracy,
   precision, recall, F1, macro-F1, AUC-ROC and loss. That file is the evidence
   for the 15 points.

## Integration

Imports `config`, `features`, `models` and `evaluation` (for the per-epoch
metric computation). Imported by the training notebook only. Must not import
`serving` or `explain`.

Done when: a synthetic run whose validation score improves for three epochs then
worsens stops after exactly `patience` further epochs, and the returned weights
match the epoch-3 checkpoint rather than the final one. Test lives in
`tests/critical/test_early_stopping.py`.
