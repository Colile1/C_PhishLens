# loading.py
# Purpose: load the vocabulary, checkpoint and validation-selected threshold once.
# Author: Colile Sibanda

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from phishlens.config.schema import Config
from phishlens.features.vocabulary import Vocabulary
from phishlens.models.text_cnn import build_text_cnn
from phishlens.training.checkpoint import restore_best

METADATA_FILENAME = "m1_text_cnn.json"
UNKNOWN_COMMIT = "unknown"


def vocabulary_hash(path: Path) -> str:
    """Hash a vocabulary file's content, for the metadata endpoint.

    Line endings are normalised to LF first. Git checks the file out with
    CRLF on Windows and LF on Linux, so hashing raw bytes gave the same
    vocabulary two different hashes on a local run and on the hosted
    service.

    Inputs: path : the vocabulary JSON file on disk.
    Output: a short SHA-256 hex digest identifying the exact vocabulary the
    loaded model was encoded with, identical on every platform.
    """
    content = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()[:16]


def load_threshold_and_provenance(metrics_path: Path) -> tuple[float, str, str]:
    """Read the validation-selected threshold and provenance off the M1 metrics file.

    The threshold served here must be the one selected on validation and
    reported against test in `04_evaluate.ipynb`, not the untuned config
    default : serving at a different operating point than the reported one
    would mean the demo shows a model that was never actually evaluated.

    Inputs: metrics_path : `artifacts/metrics/m1_text_cnn.json`, written by
    `evaluation.persistence.save_metrics` in notebooks/04_evaluate.ipynb.
    Output: (threshold, training_commit, dataset_version) tuple.

    Raises FileNotFoundError naming the missing path if the metrics file is
    absent : startup must fail loudly rather than silently serve at the
    config default 0.5, which is a different, unevaluated model.
    """
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"No metrics file at {metrics_path}. Run notebooks/04_evaluate.ipynb "
            f"first so the validation-selected threshold exists; "
            f"serving at an unevaluated default threshold is not permitted."
        )
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    threshold = float(payload["threshold"])
    commit = str(payload.get("commit_hash", UNKNOWN_COMMIT))
    config_hash = str(payload.get("config_hash", UNKNOWN_COMMIT))
    return threshold, commit, config_hash


def load_vocabulary_and_model(config: Config):
    """Load the vocabulary and restore the best M1 checkpoint into a fresh model.

    Inputs: config : validated Config supplying `paths.models` and
    `serving.model_file`.
    Output: (Vocabulary, TextCNN) tuple, the model already in eval() mode.

    Raises FileNotFoundError naming the missing vocabulary or checkpoint path.
    """
    models_dir = Path(config.paths.models)
    vocabulary = Vocabulary.load(models_dir / "vocabulary.json")

    model = build_text_cnn(vocab_size=vocabulary.size(), config=config)
    restore_best(model, Path(config.serving.model_file))
    model.eval()
    return vocabulary, model
