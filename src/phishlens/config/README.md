# src/phishlens/config/

Loads and validates the YAML configuration, and seeds every source of
randomness from the single `seed` value. Every other module takes a validated
config object as input rather than reading files or reaching for globals.

## Files

| File | Responsibility |
|------|----------------|
| `loader.py` | `load_config(path="configs/default.yaml") -> Config` : read the YAML, deep-merge `configs/local.yaml` over it when present, validate, return an immutable config object |
| `schema.py` | `Config` and its nested frozen dataclasses (paths, data, preprocessing, split, features, model, training, evaluation, explain, serving, desktop). Types live here, nowhere else |
| `validation.py` | `validate(raw: dict) -> None` : required fields present, split ratios sum to 1.0, patience and step counts are positive integers, `positive_label` is 0 or 1, the port is valid. One specific `ConfigError` message per rule, naming the field |
| `seeding.py` | `set_global_seed(seed)` seeds `random`, `numpy` and `torch` and sets deterministic flags; `make_rng(seed)` returns a `numpy.random.Generator` for code that should not rely on global state |

## Integration

Imported by everything. Imports nothing from the rest of the package: it is the
root of the dependency graph. No other module opens a YAML file, and no other
module calls `set_global_seed` more than once per run.

Verified by `tests/unit/test_config.py`: a config missing `seed` is rejected
with an error naming `seed`, ratios that do not sum to one are rejected, and two
runs with the same seed draw identical values.
