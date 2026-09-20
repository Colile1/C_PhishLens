# predict_fn.py
# Purpose: the single prediction path LIME, the API and the notebook all share.
# Author: Colile Sibanda

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
import torch
from torch import nn

from phishlens.config.schema import Config
from phishlens.features.encoder import encode
from phishlens.features.vocabulary import Vocabulary

# LIME's contract: classifier_fn takes a list of strings and returns a
# (n_samples, n_classes) array of probabilities. Column 1 is phishing, column
# 0 is legitimate, matching data.positive_label = 1 throughout the project.
LEGITIMATE_COLUMN = 0
PHISHING_COLUMN = 1


class ProbabilityFn(Protocol):
    """Callable turning a list of texts into a (n, 2) probability array."""

    def __call__(self, texts: list[str]) -> np.ndarray: ...


def _as_two_columns(phishing_probability: np.ndarray) -> np.ndarray:
    """Widen a vector of phishing probabilities into LIME's two-column form.

    Inputs: phishing_probability : 1-D array of P(phishing) in [0, 1].
    Output: (n, 2) array whose column 1 is the input and column 0 is its
    complement, so the rows sum to 1 as LIME's ridge regression expects.
    """
    phishing = np.asarray(phishing_probability, dtype=np.float64).reshape(-1)
    return np.column_stack([1.0 - phishing, phishing])


def make_text_cnn_predict_fn(
    model: nn.Module,
    vocabulary: Vocabulary,
    config: Config,
    device: torch.device | None = None,
    batch_size: int | None = None,
) -> ProbabilityFn:
    """Build the M1 probability function LIME and the API both call.

    The returned closure runs the *whole* inference path : normalise, tokenise,
    encode, forward, sigmoid : so an explanation describes the deployed
    pipeline rather than a reimplementation of it (explain/README.md).

    The model is switched to eval() and gradients are disabled: LIME calls
    this once per perturbation (num_samples times per explanation), so dropout
    left active would make the explanation non-deterministic and the local
    surrogate would fit noise.

    Inputs: model : a TextCNN with the best weights already restored;
    vocabulary : the training-set vocabulary the model was trained with;
    config : validated Config; device : where to run, defaulting to the
    model's own device; batch_size : inference batch size, defaulting to
    config.training.batch_size.
    Output: a callable taking list[str] and returning an (n, 2) float array,
    column 1 being P(phishing).
    """
    target_device = device if device is not None else next(model.parameters()).device
    size = batch_size if batch_size is not None else config.training.batch_size
    model.eval()
    model.to(target_device)

    def predict(texts: list[str]) -> np.ndarray:
        """Score a list of raw texts and return (n, 2) class probabilities."""
        if not texts:
            return np.empty((0, 2), dtype=np.float64)

        probabilities: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), size):
                chunk = list(texts[start : start + size])
                # encode() normalises internally, so callers hand it raw or
                # normalised text and get the same surface form either way:
                # normalise_text is idempotent (features/normalise.py).
                encoded = torch.from_numpy(encode(chunk, vocabulary, config))
                logits = model(encoded.to(target_device))
                probabilities.append(torch.sigmoid(logits).cpu().numpy())

        return _as_two_columns(np.concatenate(probabilities))

    return predict


def make_baseline_predict_fn(pipeline) -> ProbabilityFn:
    """Wrap a fitted M0 sklearn pipeline in the same two-column contract.

    Exists so M0 and M1 explanations are produced through interfaces of the
    same shape and are therefore comparable in `06_xai.ipynb`, which is the
    stated reason LIME was chosen as the primary method over a model-specific
    attribution.

    Inputs: pipeline : a fitted Pipeline from models.build_baseline, whose
    classifier exposes predict_proba with classes_ [0, 1].
    Output: a callable taking list[str] and returning an (n, 2) float array,
    column 1 being P(phishing).

    Raises ValueError if the pipeline's classes are not exactly [0, 1], which
    would mean column 1 is not the phishing column and every explanation sign
    would be silently inverted.
    """
    classes = list(getattr(pipeline, "classes_", [0, 1]))
    if classes != [0, 1]:
        raise ValueError(
            f"Baseline pipeline reports classes_ {classes!r}, expected [0, 1] so "
            f"that column {PHISHING_COLUMN} is the phishing probability. An "
            f"explanation from this pipeline would carry inverted signs."
        )

    def predict(texts: list[str]) -> np.ndarray:
        """Score a list of raw texts with the fitted M0 pipeline."""
        if not texts:
            return np.empty((0, 2), dtype=np.float64)
        return np.asarray(pipeline.predict_proba(list(texts)), dtype=np.float64)

    return predict
