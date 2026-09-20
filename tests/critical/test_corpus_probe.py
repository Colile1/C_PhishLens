# test_corpus_probe.py
# Purpose: measure how strongly the pooled corpora carry source-corpus markers.
# Author: Colile Sibanda

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from phishlens.config import load_config

METRICS_FILE = "corpus_probe.json"

# Bhuiyan & Bhuiyan (2026) reported 0.97-0.98 on a comparable pool. A score in
# that region is the expected finding, not a failure: it is the evidence that
# the headline random-split numbers are partly corpus recognition. This test
# therefore asserts only that the probe was built and recorded.
EXPECTED_HIGH = 0.9

# The probe is a diagnostic, not a deliverable, so it is deliberately cheap.
PROBE_MAX_FEATURES = 20000


@pytest.fixture(scope="module")
def config():
    """Load the project config once for the module.

    Inputs: none.
    Output: validated Config from configs/default.yaml.
    """
    return load_config()


@pytest.fixture(scope="module")
def split_frames(config):
    """Read the train and test split files.

    The probe trains on train and scores on test, the same partition the
    phishing models use, so its accuracy is directly comparable with theirs.
    Scoring the probe on the test split does not spend the project's
    once-only test-set budget: the probe predicts `corpus`, never `label`,
    and no modelling decision is taken from its result.

    Inputs: config : validated Config supplying paths.processed.
    Output: dict of split name to DataFrame.

    Skips the module if the split has not been built yet.
    """
    processed_dir = Path(config.paths.processed)
    frames = {}
    for split_name in ("train", "test"):
        split_path = processed_dir / f"{split_name}.csv"
        if not split_path.is_file():
            pytest.skip(
                f"Split file {split_path} is missing. Build the split first: "
                "python scripts/build_splits.py"
            )
        frames[split_name] = pd.read_csv(split_path, usecols=["text", "corpus"])
    return frames


@pytest.fixture(scope="module")
def probe_result(config, split_frames):
    """Train the source-corpus probe and return its scores.

    A TF-IDF + Logistic Regression probe predicting which corpus an email came
    from, using the same text the phishing models see. Deliberately the same
    model family as the M0 baseline: whatever signal this probe finds, M0 can
    find too.

    Inputs: config : validated Config supplying the seed; split_frames : the
    train and test frames.
    Output: dict of accuracy, per-corpus F1, macro F1 and the row counts.
    """
    train = split_frames["train"]
    test = split_frames["test"]

    vectoriser = TfidfVectorizer(
        max_features=PROBE_MAX_FEATURES,
        ngram_range=tuple(config.features.tfidf.ngram_range),
        min_df=config.features.tfidf.min_df,
    )
    train_matrix = vectoriser.fit_transform(train["text"].astype(str))
    test_matrix = vectoriser.transform(test["text"].astype(str))

    classifier = LogisticRegression(
        max_iter=1000,
        random_state=config.seed,
        class_weight="balanced",
    )
    classifier.fit(train_matrix, train["corpus"].astype(str))
    predicted = classifier.predict(test_matrix)
    actual = test["corpus"].astype(str)

    labels = sorted(set(actual) | set(predicted))
    per_corpus = f1_score(actual, predicted, labels=labels, average=None, zero_division=0)

    return {
        "accuracy": round(float(accuracy_score(actual, predicted)), 4),
        "macro_f1": round(
            float(f1_score(actual, predicted, average="macro", zero_division=0)), 4
        ),
        "per_corpus_f1": {
            corpus: round(float(score), 4)
            for corpus, score in zip(labels, per_corpus)
        },
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "seed": config.seed,
    }


def test_probe_trains_and_scores(probe_result):
    """The probe must build and produce a usable accuracy.

    The test fails only if the probe cannot be built. A high score is the
    finding this guard exists to record, so no upper bound is asserted.
    """
    accuracy = probe_result["accuracy"]
    assert 0.0 <= accuracy <= 1.0, (
        f"Corpus probe returned accuracy {accuracy}, which is not a valid "
        f"proportion. The probe did not train correctly."
    )
    assert probe_result["train_rows"] > 0 and probe_result["test_rows"] > 0


def test_probe_beats_chance(probe_result, split_frames):
    """The probe must do better than always guessing the largest corpus.

    If it did not, the probe itself would be broken and its recorded score
    would understate the artefact rather than measure it.
    """
    majority = split_frames["test"]["corpus"].astype(str).value_counts(normalize=True)
    baseline = float(majority.iloc[0])

    assert probe_result["accuracy"] > baseline, (
        f"Corpus probe scored {probe_result['accuracy']:.4f}, no better than "
        f"the majority-class baseline of {baseline:.4f}. The probe is broken, "
        f"so its score cannot be used as evidence either way."
    )


def test_probe_accuracy_is_recorded(config, probe_result):
    """The probe's score is written to artifacts/metrics/corpus_probe.json.

    The report quotes this file. If it is not written the finding is lost,
    which is the one way this guard can fail its purpose.
    """
    metrics_dir = Path(config.paths.metrics)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / METRICS_FILE

    record = dict(probe_result)
    record["interpretation"] = (
        "Accuracy of a TF-IDF + Logistic Regression probe predicting an "
        "email's source corpus from its text alone. A high score means the "
        "pooled corpora carry strong source markers, so headline random-split "
        "phishing scores are partly corpus recognition. This is a reported "
        "limitation, not a defect to fix."
    )
    record["expected_high_threshold"] = EXPECTED_HIGH
    record["above_expected_high"] = bool(probe_result["accuracy"] >= EXPECTED_HIGH)

    metrics_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    written = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert written["accuracy"] == probe_result["accuracy"], (
        f"Wrote {metrics_path} but read back a different accuracy."
    )
