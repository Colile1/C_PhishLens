# configs/

Single source of truth for every tunable value: seeds, paths, split ratios,
feature settings, model hyperparameters, training and early-stopping settings,
explanation settings, serving and rate-limit settings, and the desktop client's
API address.

No module under `src/` hard-codes a value that belongs here. If a number appears
in code and also matters to a result, it belongs in this folder.

## Files

| File | Purpose |
|------|---------|
| `default.yaml` | The committed configuration. Every reported result is reproducible from this file, the pinned `requirements.txt` and the commit hash |
| `local.yaml` | Optional machine-specific overrides (for example the desktop app's `api_base_url`, or a hosted bind address). Not committed. Deep-merged over `default.yaml` by the loader when present |

## Integration

Read by `src/phishlens/config/` only. Every other module receives an
already-validated config object, which keeps the core testable with a
hand-built config and no filesystem. The deploy images write their own
`local.yaml` (see `deploy/`).
