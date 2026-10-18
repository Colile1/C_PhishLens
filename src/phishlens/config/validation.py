# validation.py
# Purpose: validate the raw config dict before it is built into a Config object.
# Author: Colile Sibanda

from __future__ import annotations

from typing import Any

_SPLIT_RATIO_TOLERANCE = 1e-6

_REQUIRED_TOP_LEVEL = (
    "seed",
    "paths",
    "data",
    "preprocess",
    "split",
    "features",
    "model",
    "training",
    "evaluation",
    "explain",
    "serving",
    "desktop",
)

_REQUIRED_PATHS = ("raw", "interim", "processed", "models", "figures", "metrics")

_REQUIRED_DATA = (
    "corpora",
    "text_columns",
    "label_column",
    "positive_label",
    "audit_only_columns",
    "min_characters",
)

_REQUIRED_PREPROCESS = (
    "lowercase",
    "strip_html",
    "url_placeholder",
    "email_placeholder",
    "number_placeholder",
    "subject_body_separator",
)

_REQUIRED_SPLIT = ("train", "validation", "test", "stratify_by", "group_by")

_REQUIRED_FEATURES = ("tfidf", "vocabulary", "max_sequence_length")
_REQUIRED_TFIDF = ("max_features", "ngram_range", "min_df")
_REQUIRED_VOCABULARY = ("max_size", "min_frequency", "unknown_token", "padding_token")

_REQUIRED_MODEL = (
    "name",
    "embedding_dim",
    "filter_sizes",
    "filters_per_size",
    "dropout",
)

_REQUIRED_TRAINING = (
    "batch_size",
    "learning_rate",
    "max_epochs",
    "class_weighting",
    "early_stopping",
)
_REQUIRED_EARLY_STOPPING = (
    "metric",
    "mode",
    "patience",
    "min_delta",
    "restore_best_weights",
)

_REQUIRED_EVALUATION = ("decision_threshold", "figure_dpi")

_REQUIRED_EXPLAIN = ("method", "num_features", "num_samples", "stability_seeds", "ig_steps")

_REQUIRED_SERVING = (
    "host", "port", "model_file", "max_input_characters", "rate_limit_per_minute",
    "explain_rate_limit_per_minute", "explain_global_limit_per_minute",
    "max_concurrent_requests", "trusted_proxy_hops",
)

_REQUIRED_DESKTOP = ("api_base_url", "request_timeout_seconds")


class ConfigError(ValueError):
    """Raised when the raw configuration fails validation."""


def _require_fields(section: dict[str, Any], fields: tuple[str, ...], section_name: str) -> None:
    """Raise ConfigError naming the first missing field in a section.

    Inputs: the section dict, the tuple of required field names, the section's
    dotted name used in the error message.
    Output: None. Raises ConfigError if a field is missing.
    """
    for field_name in fields:
        if field_name not in section:
            raise ConfigError(
                f"Config field '{section_name}.{field_name}' is missing."
            )


def validate(raw: dict[str, Any]) -> None:
    """Validate the raw config dict loaded from YAML.

    Inputs: raw, the dict produced by yaml.safe_load on the merged config.
    Output: None. Raises ConfigError naming the offending field or rule on any
    failure. Does not mutate raw.
    """
    if not isinstance(raw, dict):
        raise ConfigError("Config root must be a mapping.")

    _require_fields(raw, _REQUIRED_TOP_LEVEL, "config")

    if not isinstance(raw["seed"], int) or isinstance(raw["seed"], bool):
        raise ConfigError("Config field 'seed' must be an integer.")

    _require_fields(raw["paths"], _REQUIRED_PATHS, "paths")
    for name in _REQUIRED_PATHS:
        if not raw["paths"][name]:
            raise ConfigError(f"Config field 'paths.{name}' must be non-empty.")

    _require_fields(raw["data"], _REQUIRED_DATA, "data")
    if raw["data"]["positive_label"] not in (0, 1):
        raise ConfigError(
            "Config field 'data.positive_label' must be 0 or 1, got "
            f"{raw['data']['positive_label']!r}."
        )
    if raw["data"]["min_characters"] < 0:
        raise ConfigError("Config field 'data.min_characters' must be >= 0.")

    _require_fields(raw["preprocess"], _REQUIRED_PREPROCESS, "preprocess")

    _require_fields(raw["split"], _REQUIRED_SPLIT, "split")
    ratio_sum = raw["split"]["train"] + raw["split"]["validation"] + raw["split"]["test"]
    if abs(ratio_sum - 1.0) > _SPLIT_RATIO_TOLERANCE:
        raise ConfigError(
            "Config fields 'split.train' + 'split.validation' + 'split.test' "
            f"must sum to 1.0, got {ratio_sum}."
        )
    for name in ("train", "validation", "test"):
        if not (0.0 < raw["split"][name] < 1.0):
            raise ConfigError(f"Config field 'split.{name}' must be between 0 and 1.")

    _require_fields(raw["features"], _REQUIRED_FEATURES, "features")
    _require_fields(raw["features"]["tfidf"], _REQUIRED_TFIDF, "features.tfidf")
    _require_fields(
        raw["features"]["vocabulary"], _REQUIRED_VOCABULARY, "features.vocabulary"
    )

    _require_fields(raw["model"], _REQUIRED_MODEL, "model")
    if not (0.0 <= raw["model"]["dropout"] < 1.0):
        raise ConfigError("Config field 'model.dropout' must be in [0, 1).")

    _require_fields(raw["training"], _REQUIRED_TRAINING, "training")
    _require_fields(
        raw["training"]["early_stopping"],
        _REQUIRED_EARLY_STOPPING,
        "training.early_stopping",
    )
    patience = raw["training"]["early_stopping"]["patience"]
    if not isinstance(patience, int) or isinstance(patience, bool) or patience <= 0:
        raise ConfigError(
            "Config field 'training.early_stopping.patience' must be a positive integer, "
            f"got {patience!r}."
        )
    mode = raw["training"]["early_stopping"]["mode"]
    if mode not in ("max", "min"):
        raise ConfigError(
            f"Config field 'training.early_stopping.mode' must be 'max' or 'min', got {mode!r}."
        )

    _require_fields(raw["evaluation"], _REQUIRED_EVALUATION, "evaluation")
    if not (0.0 <= raw["evaluation"]["decision_threshold"] <= 1.0):
        raise ConfigError(
            "Config field 'evaluation.decision_threshold' must be between 0 and 1."
        )

    _require_fields(raw["explain"], _REQUIRED_EXPLAIN, "explain")
    ig_steps = raw["explain"]["ig_steps"]
    if not isinstance(ig_steps, int) or isinstance(ig_steps, bool) or ig_steps <= 0:
        raise ConfigError(
            f"Config field 'explain.ig_steps' must be a positive integer, got {ig_steps!r}."
        )

    _require_fields(raw["serving"], _REQUIRED_SERVING, "serving")
    if not (0 < raw["serving"]["port"] < 65536):
        raise ConfigError("Config field 'serving.port' must be a valid TCP port.")
    for name in ("rate_limit_per_minute", "explain_rate_limit_per_minute",
                 "explain_global_limit_per_minute", "max_concurrent_requests"):
        value = raw["serving"][name]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ConfigError(f"Config field 'serving.{name}' must be a positive integer, got {value!r}.")
    hops = raw["serving"]["trusted_proxy_hops"]
    if not isinstance(hops, int) or isinstance(hops, bool) or hops < 0:
        raise ConfigError(f"Config field 'serving.trusted_proxy_hops' must be 0 or more, got {hops!r}.")

    _require_fields(raw["desktop"], _REQUIRED_DESKTOP, "desktop")
    if not str(raw["desktop"]["api_base_url"]).strip():
        raise ConfigError("Config field 'desktop.api_base_url' must be non-empty.")
    timeout = raw["desktop"]["request_timeout_seconds"]
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
        raise ConfigError(
            "Config field 'desktop.request_timeout_seconds' must be a positive number, "
            f"got {timeout!r}."
        )
