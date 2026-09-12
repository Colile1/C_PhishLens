# configs/

Single source of truth for every tunable value: seeds, paths, split ratios,
feature settings, model hyperparameters, training and early-stopping settings,
explanation settings and serving settings.

No module under `src/` may hard-code a value that belongs here. If a number
appears in code and also matters to a result, it belongs in this folder.

## Planned files

| File | Responsibility |
|------|----------------|
| `default.yaml` | The committed configuration. Every reported result must be reproducible from this file plus the pinned `requirements.txt` and the commit hash |
| `local.yaml` | Optional machine-specific overrides (paths, device). Git-ignored. Merged over `default.yaml` by the loader when present |
| `smoke.yaml` | Tiny configuration for the end-to-end smoke run: a few thousand rows, two epochs. Added |

## Integration

Read by `src/phishlens/config/` only. Nothing else opens these files directly —
every other module receives an already-validated config object. That keeps the
core testable with a hand-built config and no filesystem.

Done when: the loader rejects `default.yaml` with a missing `seed` by raising an
error that names `seed`, and two runs with the same seed produce identical
splits and identical first-batch draws.
