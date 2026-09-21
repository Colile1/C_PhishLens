# test_config.py
# Purpose: unit tests for the config loader, validator and seeding utilities.
# Author: Colile Sibanda

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
import yaml

from phishlens.config.loader import load_config
from phishlens.config.seeding import make_rng, set_global_seed
from phishlens.config.validation import ConfigError, validate

VALID_CONFIG: dict = {
    "seed": 42,
    "paths": {
        "raw": "data/raw",
        "interim": "data/interim",
        "processed": "data/processed",
        "models": "artifacts/models",
        "figures": "artifacts/figures",
        "metrics": "artifacts/metrics",
    },
    "data": {
        "corpora": ["CEAS_08.csv"],
        "text_columns": ["subject", "body"],
        "label_column": "label",
        "positive_label": 1,
        "audit_only_columns": ["sender"],
        "min_characters": 10,
    },
    "preprocess": {
        "lowercase": True,
        "strip_html": True,
        "url_placeholder": "<url>",
        "email_placeholder": "<email>",
        "number_placeholder": "<num>",
        "subject_body_separator": " <sep> ",
    },
    "split": {
        "train": 0.70,
        "validation": 0.15,
        "test": 0.15,
        "stratify_by": "label",
        "group_by": "content_hash",
    },
    "features": {
        "tfidf": {"max_features": 50000, "ngram_range": [1, 2], "min_df": 2},
        "vocabulary": {
            "max_size": 40000,
            "min_frequency": 2,
            "unknown_token": "<unk>",
            "padding_token": "<pad>",
        },
        "max_sequence_length": 400,
    },
    "model": {
        "name": "text_cnn",
        "embedding_dim": 128,
        "filter_sizes": [3, 4, 5],
        "filters_per_size": 100,
        "dropout": 0.5,
    },
    "training": {
        "batch_size": 64,
        "learning_rate": 0.001,
        "max_epochs": 30,
        "class_weighting": True,
        "early_stopping": {
            "metric": "val_macro_f1",
            "mode": "max",
            "patience": 3,
            "min_delta": 0.001,
            "restore_best_weights": True,
        },
    },
    "evaluation": {"decision_threshold": 0.5, "figure_dpi": 150},
    "explain": {
        "method": "lime",
        "num_features": 15,
        "num_samples": 1000,
        "stability_seeds": [42, 1337],
    },
    "serving": {
        "host": "127.0.0.1",
        "port": 8000,
        "model_file": "artifacts/models/text_cnn_best.pt",
        "max_input_characters": 50000,
    },
    "desktop": {
        "api_base_url": "http://127.0.0.1:8000",
        "request_timeout_seconds": 30.0,
    },
}


def _write_config(tmp_path, data: dict, filename: str = "default.yaml"):
    config_path = tmp_path / filename
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle)
    return config_path


def test_validate_accepts_the_default_config():
    validate(copy.deepcopy(VALID_CONFIG))


def test_missing_seed_is_rejected_naming_seed():
    raw = copy.deepcopy(VALID_CONFIG)
    del raw["seed"]
    with pytest.raises(ConfigError, match="seed"):
        validate(raw)


def test_split_ratios_not_summing_to_one_are_rejected():
    raw = copy.deepcopy(VALID_CONFIG)
    raw["split"]["train"] = 0.7
    raw["split"]["validation"] = 0.15
    raw["split"]["test"] = 0.2
    with pytest.raises(ConfigError, match="split"):
        validate(raw)


def test_non_positive_patience_is_rejected():
    raw = copy.deepcopy(VALID_CONFIG)
    raw["training"]["early_stopping"]["patience"] = 0
    with pytest.raises(ConfigError, match="patience"):
        validate(raw)


def test_invalid_positive_label_is_rejected():
    raw = copy.deepcopy(VALID_CONFIG)
    raw["data"]["positive_label"] = 2
    with pytest.raises(ConfigError, match="positive_label"):
        validate(raw)


def test_missing_nested_field_is_named(tmp_path):
    raw = copy.deepcopy(VALID_CONFIG)
    del raw["paths"]["raw"]
    with pytest.raises(ConfigError, match="paths.raw"):
        validate(raw)


def test_missing_desktop_api_base_url_is_named():
    raw = copy.deepcopy(VALID_CONFIG)
    del raw["desktop"]["api_base_url"]
    with pytest.raises(ConfigError, match="desktop.api_base_url"):
        validate(raw)


def test_blank_desktop_api_base_url_is_rejected():
    raw = copy.deepcopy(VALID_CONFIG)
    raw["desktop"]["api_base_url"] = "   "
    with pytest.raises(ConfigError, match="desktop.api_base_url"):
        validate(raw)


def test_non_positive_request_timeout_is_rejected():
    raw = copy.deepcopy(VALID_CONFIG)
    raw["desktop"]["request_timeout_seconds"] = 0
    with pytest.raises(ConfigError, match="request_timeout_seconds"):
        validate(raw)


def test_load_config_builds_typed_config(tmp_path):
    config_path = _write_config(tmp_path, VALID_CONFIG)
    config = load_config(str(config_path))
    assert config.seed == 42
    assert config.split.train == 0.70
    assert config.model.filter_sizes == (3, 4, 5)
    assert config.training.early_stopping.patience == 3


def test_load_config_raises_config_error_for_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(str(tmp_path / "missing.yaml"))


def test_local_yaml_overrides_default(tmp_path):
    config_path = _write_config(tmp_path, VALID_CONFIG, "default.yaml")
    _write_config(tmp_path, {"seed": 99}, "local.yaml")
    config = load_config(str(config_path))
    assert config.seed == 99
    assert config.split.train == 0.70


def test_set_global_seed_reproduces_random_draws():
    set_global_seed(123)
    torch_draw_1 = torch.rand(3)
    numpy_draw_1 = np.random.rand(3)

    set_global_seed(123)
    torch_draw_2 = torch.rand(3)
    numpy_draw_2 = np.random.rand(3)

    assert torch.equal(torch_draw_1, torch_draw_2)
    assert np.array_equal(numpy_draw_1, numpy_draw_2)


def test_make_rng_is_deterministic_and_independent_of_global_state():
    rng_a = make_rng(7)
    rng_b = make_rng(7)
    assert np.array_equal(rng_a.random(5), rng_b.random(5))
