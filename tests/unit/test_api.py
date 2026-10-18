# test_api.py
# Purpose: unit tests for the FastAPI serving layer, via TestClient.
# Author: Colile Sibanda

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from phishlens.config.loader import load_config
from phishlens.serving.app import create_app

PHISHING_LURE = (
    "URGENT: your account has been suspended. Verify your password immediately "
    "at http://secure-update.example/login or your account will be closed "
    "within 24 hours. Click here to confirm your identity now."
)


@pytest.fixture(scope="module")
def client():
    """Build a TestClient wrapping the real app, model and checkpoint.

    `TestClient` used as a context manager runs the app's lifespan, so
    startup actually loads the vocabulary, checkpoint and threshold once for
    every test in this module, matching how `run_api.py` serves it.

    Inputs: none.
    Output: a live TestClient with the model loaded.
    """
    config = load_config()
    app = create_app(config)
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_model_loaded(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_metadata_reports_provenance(client: TestClient) -> None:
    response = client.get("/metadata")
    assert response.status_code == 200
    body = response.json()
    assert body["model_id"] == "m1_text_cnn"
    assert body["architecture"] == "TextCNN"
    assert 0.0 <= body["threshold"] <= 1.0
    assert body["seed"] == 42
    assert body["training_commit"]
    assert body["vocabulary_hash"]
    assert body["dataset_version"]


def test_predict_returns_probability_in_range(client: TestClient) -> None:
    response = client.post("/predict", json={"text": PHISHING_LURE})
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["label"] in ("phishing", "legitimate")
    assert body["model_id"] == "m1_text_cnn"
    assert 0.0 <= body["threshold"] <= 1.0


def test_predict_labels_a_known_phishing_lure_as_phishing(client: TestClient) -> None:
    response = client.post("/predict", json={"text": PHISHING_LURE})
    assert response.status_code == 200
    assert response.json()["label"] == "phishing"


def test_predict_empty_text_returns_422_naming_text(client: TestClient) -> None:
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 422
    body = response.json()
    fields = [entry["loc"][-1] for entry in body["detail"]]
    assert "text" in fields


def test_predict_missing_text_field_returns_422(client: TestClient) -> None:
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_over_length_returns_413(client: TestClient) -> None:
    config = load_config()
    oversized = "a" * (config.serving.max_input_characters + 1)
    response = client.post("/predict", json={"text": oversized})
    assert response.status_code == 413


def test_explain_offsets_index_into_the_submitted_text(client: TestClient) -> None:
    response = client.post("/explain", json={"text": PHISHING_LURE})
    assert response.status_code == 200
    body = response.json()
    assert body["method"] == "lime"
    assert "local" in body["caveat"].lower()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["tokens"], "expected at least one token in the explanation"
    assert body["text"], "expected the normalised text the offsets index into"

    # Every reported token must slice back out of `text` at its own offsets:
    # this is the design contract the desktop app's highlighting depends on.
    for entry in body["tokens"]:
        assert entry["end"] > entry["start"]
        assert body["text"][entry["start"] : entry["end"]] == entry["token"]

    # Offsets are non-decreasing across the token list, i.e. tokens are
    # reported in reading order, not scrambled by weight.
    starts = [entry["start"] for entry in body["tokens"]]
    assert starts == sorted(starts)


def test_explain_empty_text_returns_422_naming_text(client: TestClient) -> None:
    response = client.post("/explain", json={"text": ""})
    assert response.status_code == 422
    body = response.json()
    fields = [entry["loc"][-1] for entry in body["detail"]]
    assert "text" in fields


def test_vocabulary_hash_ignores_line_endings(tmp_path) -> None:
    from phishlens.serving.loading import vocabulary_hash

    lf = tmp_path / "lf.json"
    crlf = tmp_path / "crlf.json"
    lf.write_bytes(b'{\n  "a": 1\n}\n')
    crlf.write_bytes(b'{\r\n  "a": 1\r\n}\r\n')
    assert vocabulary_hash(lf) == vocabulary_hash(crlf)
