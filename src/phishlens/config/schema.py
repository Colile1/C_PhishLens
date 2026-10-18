# schema.py
# Purpose: frozen dataclasses describing the validated configuration.
# Author: Colile Sibanda

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem locations used across the pipeline.

    Inputs: raw, interim, processed, models, figures, metrics directory paths.
    Output: immutable PathsConfig.
    """

    raw: str
    interim: str
    processed: str
    models: str
    figures: str
    metrics: str


@dataclass(frozen=True)
class DataConfig:
    """Corpus files and column conventions for the pooled dataset.

    Inputs: corpus file list, text/label columns, positive label, audit-only
    columns, minimum row length.
    Output: immutable DataConfig.
    """

    corpora: tuple[str, ...]
    text_columns: tuple[str, ...]
    label_column: str
    positive_label: int
    audit_only_columns: tuple[str, ...]
    min_characters: int


@dataclass(frozen=True)
class PreprocessConfig:
    """Text normalisation switches and placeholder tokens.

    Inputs: lowercase/strip_html flags, url/email/number placeholders,
    subject-body separator.
    Output: immutable PreprocessConfig.
    """

    lowercase: bool
    strip_html: bool
    url_placeholder: str
    email_placeholder: str
    number_placeholder: str
    subject_body_separator: str


@dataclass(frozen=True)
class SplitConfig:
    """Train/validation/test ratios and the grouping key for splitting.

    Inputs: train, validation, test ratios; stratify_by and group_by columns.
    Output: immutable SplitConfig.
    """

    train: float
    validation: float
    test: float
    stratify_by: str
    group_by: str


@dataclass(frozen=True)
class TfidfConfig:
    """TF-IDF vectoriser settings for the M0 baseline."""

    max_features: int
    ngram_range: tuple[int, int]
    min_df: int


@dataclass(frozen=True)
class VocabularyConfig:
    """Vocabulary and integer-encoder settings for the M1 Text-CNN."""

    max_size: int
    min_frequency: int
    unknown_token: str
    padding_token: str


@dataclass(frozen=True)
class FeaturesConfig:
    """Feature-layer configuration for both M0 and M1.

    Inputs: tfidf settings, vocabulary settings, max sequence length.
    Output: immutable FeaturesConfig.
    """

    tfidf: TfidfConfig
    vocabulary: VocabularyConfig
    max_sequence_length: int


@dataclass(frozen=True)
class ModelConfig:
    """Text-CNN architecture hyperparameters.

    Inputs: model name, embedding dim, filter sizes, filters per size, dropout.
    Output: immutable ModelConfig.
    """

    name: str
    embedding_dim: int
    filter_sizes: tuple[int, ...]
    filters_per_size: int
    dropout: float


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Patience-based early stopping settings for the training loop."""

    metric: str
    mode: str
    patience: int
    min_delta: float
    restore_best_weights: bool


@dataclass(frozen=True)
class TrainingConfig:
    """Training loop hyperparameters and early-stopping configuration.

    Inputs: batch size, learning rate, max epochs, class weighting flag,
    early stopping settings.
    Output: immutable TrainingConfig.
    """

    batch_size: int
    learning_rate: float
    max_epochs: int
    class_weighting: bool
    early_stopping: EarlyStoppingConfig


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation-layer configuration.

    Inputs: decision threshold, figure DPI.
    Output: immutable EvalConfig.
    """

    decision_threshold: float
    figure_dpi: int


@dataclass(frozen=True)
class ExplainConfig:
    """XAI configuration for the LIME wrapper and stability check.

    Inputs: method name, tokens shown per explanation, LIME sample count,
    stability-check seeds, Integrated Gradients path steps.
    Output: immutable ExplainConfig.
    """

    method: str
    num_features: int
    num_samples: int
    stability_seeds: tuple[int, ...]
    ig_steps: int


@dataclass(frozen=True)
class ServingConfig:
    """FastAPI serving configuration.

    Inputs: host, port, model file path, max accepted input length, and the
    abuse limits: per-client request rates, a global /explain rate, the
    concurrent-connection cap, and how many reverse proxies to trust when
    reading the client address from X-Forwarded-For.
    Output: immutable ServingConfig.
    """

    host: str
    port: int
    model_file: str
    max_input_characters: int
    rate_limit_per_minute: int
    explain_rate_limit_per_minute: int
    explain_global_limit_per_minute: int
    max_concurrent_requests: int
    trusted_proxy_hops: int


@dataclass(frozen=True)
class DesktopConfig:
    """Tkinter desktop client configuration.

    Inputs: api_base_url, the base URL of the running FastAPI service;
    request_timeout_seconds, how long a request may take before the client
    gives up and reports the service as unreachable.
    Output: immutable DesktopConfig.
    """

    api_base_url: str
    request_timeout_seconds: float


@dataclass(frozen=True)
class Config:
    """Root configuration object passed to every module in the pipeline.

    Inputs: seed plus every nested section below.
    Output: immutable Config, the single object the rest of the codebase
    receives instead of reading YAML directly.
    """

    seed: int
    paths: PathsConfig
    data: DataConfig
    preprocess: PreprocessConfig
    split: SplitConfig
    features: FeaturesConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvalConfig
    explain: ExplainConfig
    serving: ServingConfig
    desktop: DesktopConfig
