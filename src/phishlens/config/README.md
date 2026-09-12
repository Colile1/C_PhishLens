# src/phishlens/config/

Loads and validates the YAML configuration, and seeds every source of
randomness from the single `seed` value. This is the first module built
 because every later module takes a validated config object as input
rather than reading files or reaching for globals.

## Planned files

| File | Responsibility |
|------|----------------|
| `loader.py` | `load_config(path: str = "configs/default.yaml") -> Config` — read the YAML, merge `configs/local.yaml` over it when present, validate, return an immutable config object. Raises `ConfigError` naming the offending field |
| `schema.py` | `Config` dataclass and the nested sections (`DataConfig`, `SplitConfig`, `ModelConfig`, `TrainingConfig`, `EvalConfig`, `ExplainConfig`, `ServingConfig`). Types and defaults live here, nowhere else |
| `validation.py` | `validate(raw: dict) -> None` — required fields present, split ratios sum to 1.0 within tolerance, patience is a positive integer, paths are non-empty, `positive_label` is one of the label values. One specific error message per rule |
| `seeding.py` | `set_global_seed(seed: int) -> None` — seed `random`, `numpy` and `torch` (CPU and CUDA), and set deterministic flags. `make_rng(seed: int) -> numpy.random.Generator` for code that should take a generator rather than rely on global state |

## Integration

Imported by everything. Imports nothing from the rest of the package — this is
the root of the dependency graph and must stay that way.

The rest of the codebase receives a `Config` object. No other module opens a
YAML file, and no other module calls `set_global_seed` more than once per run.

Done when: a config missing `seed` is rejected with an error whose message
contains `seed`; split ratios of 0.7/0.15/0.2 are rejected naming the sum; and
two runs with the same seed produce an identical first batch and an identical
split.
