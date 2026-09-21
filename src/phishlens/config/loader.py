# loader.py
# Purpose: load, merge, validate and build the immutable Config object.
# Author: Colile Sibanda

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from phishlens.config.schema import (
    Config,
    DataConfig,
    DesktopConfig,
    EarlyStoppingConfig,
    EvalConfig,
    ExplainConfig,
    FeaturesConfig,
    ModelConfig,
    PathsConfig,
    PreprocessConfig,
    ServingConfig,
    SplitConfig,
    TfidfConfig,
    TrainingConfig,
    VocabularyConfig,
)
from phishlens.config.validation import ConfigError, validate

_LOCAL_OVERRIDE_FILENAME = "local.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into a copy of base.

    Inputs: base, the default config dict; override, the local overrides.
    Output: a new dict with override's values layered on top of base's.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read and parse a YAML file into a dict.

    Inputs: path, the file to read.
    Output: the parsed dict. Raises ConfigError if the file is missing or does
    not parse to a mapping.
    """
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        raise ConfigError(f"Config file is empty: {path}")
    if not isinstance(data, dict):
        raise ConfigError(f"Config file must contain a mapping at the top level: {path}")
    return data


def _build_config(raw: dict[str, Any]) -> Config:
    """Build the immutable Config object from a validated raw dict.

    Inputs: raw, a dict that has already passed validate().
    Output: a Config instance.
    """
    paths = PathsConfig(**raw["paths"])

    data_raw = raw["data"]
    data = DataConfig(
        corpora=tuple(data_raw["corpora"]),
        text_columns=tuple(data_raw["text_columns"]),
        label_column=data_raw["label_column"],
        positive_label=data_raw["positive_label"],
        audit_only_columns=tuple(data_raw["audit_only_columns"]),
        min_characters=data_raw["min_characters"],
    )

    preprocess = PreprocessConfig(**raw["preprocess"])
    split = SplitConfig(**raw["split"])

    features_raw = raw["features"]
    tfidf_raw = features_raw["tfidf"]
    tfidf = TfidfConfig(
        max_features=tfidf_raw["max_features"],
        ngram_range=tuple(tfidf_raw["ngram_range"]),
        min_df=tfidf_raw["min_df"],
    )
    vocabulary = VocabularyConfig(**features_raw["vocabulary"])
    features = FeaturesConfig(
        tfidf=tfidf,
        vocabulary=vocabulary,
        max_sequence_length=features_raw["max_sequence_length"],
    )

    model_raw = raw["model"]
    model = ModelConfig(
        name=model_raw["name"],
        embedding_dim=model_raw["embedding_dim"],
        filter_sizes=tuple(model_raw["filter_sizes"]),
        filters_per_size=model_raw["filters_per_size"],
        dropout=model_raw["dropout"],
    )

    training_raw = raw["training"]
    early_stopping = EarlyStoppingConfig(**training_raw["early_stopping"])
    training = TrainingConfig(
        batch_size=training_raw["batch_size"],
        learning_rate=training_raw["learning_rate"],
        max_epochs=training_raw["max_epochs"],
        class_weighting=training_raw["class_weighting"],
        early_stopping=early_stopping,
    )

    evaluation = EvalConfig(**raw["evaluation"])

    explain_raw = raw["explain"]
    explain = ExplainConfig(
        method=explain_raw["method"],
        num_features=explain_raw["num_features"],
        num_samples=explain_raw["num_samples"],
        stability_seeds=tuple(explain_raw["stability_seeds"]),
    )

    serving = ServingConfig(**raw["serving"])

    desktop_raw = raw["desktop"]
    desktop = DesktopConfig(
        api_base_url=desktop_raw["api_base_url"],
        request_timeout_seconds=float(desktop_raw["request_timeout_seconds"]),
    )

    return Config(
        seed=raw["seed"],
        paths=paths,
        data=data,
        preprocess=preprocess,
        split=split,
        features=features,
        model=model,
        training=training,
        evaluation=evaluation,
        explain=explain,
        serving=serving,
        desktop=desktop,
    )


def load_config(path: str = "configs/default.yaml") -> Config:
    """Load, merge, validate and return the project configuration.

    Inputs: path, the default config file. When a sibling `local.yaml` exists
    in the same directory, it is deep-merged over the default.
    Output: an immutable Config object. Raises ConfigError naming the
    offending field on any validation failure.
    """
    config_path = Path(path)
    raw = _read_yaml(config_path)

    local_path = config_path.parent / _LOCAL_OVERRIDE_FILENAME
    if local_path.exists():
        raw = _deep_merge(raw, _read_yaml(local_path))

    validate(raw)
    return _build_config(raw)
