# model_service.py
# Purpose: hold the loaded M1 model, vocabulary and threshold; serve predictions and explanations.
# Author: Colile Sibanda

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from phishlens.config.schema import Config
from phishlens.explain.attributions import TokenWeight
from phishlens.explain.lime_explainer import explain_text
from phishlens.explain.predict_fn import PHISHING_COLUMN, make_text_cnn_predict_fn
from phishlens.serving.errors import EmptyTextError, TextTooLongError
from phishlens.serving.loading import (
    METADATA_FILENAME,
    load_threshold_and_provenance,
    load_vocabulary_and_model,
    vocabulary_hash,
)

MODEL_ID = "m1_text_cnn"
ARCHITECTURE = "TextCNN"

PHISHING_LABEL = "phishing"
LEGITIMATE_LABEL = "legitimate"


@dataclass(frozen=True)
class Prediction:
    """One prediction, with the decision label already derived from it.

    Inputs: label : "phishing" or "legitimate"; probability : P(phishing) in
    [0, 1]; threshold : the decision threshold `label` was derived from.
    Output: immutable Prediction.
    """

    label: str
    probability: float
    threshold: float


class ModelService:
    """Holds the loaded M1 model, vocabulary and threshold for the API.

    The single place `phishlens.models` and `phishlens.explain` are called
    from within the serving layer (serving/README.md). Constructed once by
    `load_model_service`, in `app.py`'s startup handler, then reused for
    every request.

    Inputs: config : validated Config; model : a TextCNN with best weights
    restored; vocabulary : the training-set vocabulary; threshold : the
    validation-selected decision threshold; training_commit, dataset_version,
    vocab_hash : provenance fields for `/metadata`.
    Output: a ModelService exposing `predict` and `explain`.
    """

    def __init__(
        self,
        config: Config,
        model,
        vocabulary,
        threshold: float,
        training_commit: str,
        dataset_version: str,
        vocab_hash: str,
    ) -> None:
        self._config = config
        self._model = model
        self._vocabulary = vocabulary
        self._threshold = threshold
        self._training_commit = training_commit
        self._dataset_version = dataset_version
        self._vocabulary_hash = vocab_hash
        self._predict_fn = make_text_cnn_predict_fn(model, vocabulary, config)

    @property
    def seed(self) -> int:
        """Return the config seed the served model was built under."""
        return self._config.seed

    @property
    def explain_method(self) -> str:
        """Return the configured XAI method name, e.g. "lime"."""
        return self._config.explain.method

    @property
    def threshold(self) -> float:
        """Return the decision threshold predictions are labelled at."""
        return self._threshold

    @property
    def training_commit(self) -> str:
        """Return the commit hash the served checkpoint was trained at."""
        return self._training_commit

    @property
    def dataset_version(self) -> str:
        """Return the config hash of the split the served checkpoint was trained on."""
        return self._dataset_version

    @property
    def vocabulary_hash(self) -> str:
        """Return the short hash of the served vocabulary file."""
        return self._vocabulary_hash

    def _validate_text(self, text: str) -> str:
        """Enforce the blank-text and max-length rules shared by predict and explain.

        Inputs: text : the raw submitted text.
        Output: the text unchanged, if valid.

        Raises EmptyTextError if blank or whitespace-only; TextTooLongError if
        longer than `config.serving.max_input_characters`.
        """
        if not text.strip():
            raise EmptyTextError("text must not be blank or whitespace-only.")
        limit = self._config.serving.max_input_characters
        if len(text) > limit:
            raise TextTooLongError(length=len(text), limit=limit)
        return text

    def predict(self, text: str) -> Prediction:
        """Score one email and derive its label from the serving threshold.

        Inputs: text : raw email subject and/or body text.
        Output: a Prediction with probability in [0, 1] and a label.

        Raises EmptyTextError or TextTooLongError; see `_validate_text`.
        """
        validated = self._validate_text(text)
        probabilities = self._predict_fn([validated])
        probability = float(probabilities[0, PHISHING_COLUMN])
        label = PHISHING_LABEL if probability >= self._threshold else LEGITIMATE_LABEL
        return Prediction(label=label, probability=probability, threshold=self._threshold)

    def explain(self, text: str) -> tuple[Prediction, str, list[TokenWeight]]:
        """Score and explain one email through the shared LIME predict path.

        Inputs: text : raw email subject and/or body text.
        Output: (Prediction, normalised text, token weights) : the same
        prediction `predict` would return, the normalised text LIME actually
        explained, and per-token attribution weights whose offsets
        index into that text.

        Raises EmptyTextError or TextTooLongError; see `_validate_text`.
        """
        validated = self._validate_text(text)
        explanation = explain_text(validated, self._predict_fn, self._config)
        label = (
            PHISHING_LABEL if explanation.probability >= self._threshold else LEGITIMATE_LABEL
        )
        prediction = Prediction(
            label=label, probability=explanation.probability, threshold=self._threshold
        )
        return prediction, explanation.text, explanation.tokens


def load_model_service(config: Config) -> ModelService:
    """Load the vocabulary, checkpoint and threshold once and build a ModelService.

    Called from `app.py`'s startup handler, never per request (serving/README.md).

    Inputs: config : validated Config supplying `paths.models` and
    `serving.model_file`.
    Output: a ready ModelService.

    Raises FileNotFoundError naming the missing artefact if the vocabulary,
    checkpoint or metrics file is absent.
    """
    vocabulary, model = load_vocabulary_and_model(config)
    metrics_path = Path(config.paths.metrics) / METADATA_FILENAME
    threshold, training_commit, dataset_version = load_threshold_and_provenance(metrics_path)

    return ModelService(
        config=config,
        model=model,
        vocabulary=vocabulary,
        threshold=threshold,
        training_commit=training_commit,
        dataset_version=dataset_version,
        vocab_hash=vocabulary_hash(Path(config.paths.models) / "vocabulary.json"),
    )
