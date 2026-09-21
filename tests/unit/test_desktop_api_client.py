# test_desktop_api_client.py
# Purpose: unit tests for the desktop app's HTTP client and its error messages.
# Author: Colile Sibanda

from __future__ import annotations

import pytest
import requests

from desktop import api_client
from desktop.api_client import ApiClient, ApiError

BASE_URL = "http://127.0.0.1:8000"

EXPLAIN_BODY = {
    "label": "phishing",
    "probability": 0.9731,
    "threshold": 0.5698,
    "model_id": "m1_text_cnn",
    "text": "verify your account at <url> now",
    "tokens": [
        {"token": "verify", "weight": 0.21, "start": 0, "end": 6},
        {"token": "your", "weight": 0.0, "start": 7, "end": 11},
        {"token": "account", "weight": 0.14, "start": 12, "end": 19},
        {"token": "at", "weight": -0.03, "start": 20, "end": 22},
        {"token": "<url>", "weight": 0.18, "start": 23, "end": 28},
        {"token": "now", "weight": 0.05, "start": 29, "end": 32},
    ],
    "method": "lime",
    "caveat": "This explanation is local to this prediction.",
}


class FakeResponse:
    """Stand-in for a requests.Response, carrying a status and a JSON body.

    Inputs: status_code; payload : the object `.json()` returns; text : the
    raw body, used when `payload` is the sentinel for undecodable JSON.
    Output: an object exposing the attributes ApiClient reads.
    """

    _UNDECODABLE = object()

    def __init__(self, status_code: int, payload=None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is self._UNDECODABLE:
            raise ValueError("no JSON object could be decoded")
        return self._payload


def _patch_post(monkeypatch, result) -> dict:
    """Replace requests.post with a stub returning or raising `result`.

    Inputs: monkeypatch : the pytest fixture; result : a FakeResponse to
    return, or an Exception instance to raise.
    Output: a dict the stub records the call's url/json/timeout into.
    """
    recorded: dict = {}

    def fake_post(url, json=None, timeout=None):
        recorded["url"] = url
        recorded["json"] = json
        recorded["timeout"] = timeout
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(api_client.requests, "post", fake_post)
    return recorded


def test_explain_parses_a_successful_response(monkeypatch):
    recorded = _patch_post(monkeypatch, FakeResponse(200, EXPLAIN_BODY))
    result = ApiClient(BASE_URL, 30.0).explain("verify your account")

    assert recorded["url"] == f"{BASE_URL}/explain"
    assert recorded["json"] == {"text": "verify your account"}
    assert recorded["timeout"] == 30.0
    assert result.label == "phishing"
    assert result.probability == pytest.approx(0.9731)
    assert result.threshold == pytest.approx(0.5698)
    assert result.model_id == "m1_text_cnn"
    assert result.text == EXPLAIN_BODY["text"]
    assert result.method == "lime"
    assert len(result.tokens) == 6


def test_explain_token_offsets_index_into_the_returned_text(monkeypatch):
    """The offsets must slice `text` back to their own token.

    The same alignment contract tests/critical/test_attribution_alignment.py
    enforces on the core: if the client mangled it, the app would highlight
    the wrong words with confident-looking colours.
    """
    _patch_post(monkeypatch, FakeResponse(200, EXPLAIN_BODY))
    result = ApiClient(BASE_URL, 30.0).explain("verify your account")

    for token in result.tokens:
        assert result.text[token.start : token.end] == token.token


def test_trailing_slash_in_base_url_does_not_double_up(monkeypatch):
    recorded = _patch_post(monkeypatch, FakeResponse(200, EXPLAIN_BODY))
    ApiClient(f"{BASE_URL}/", 30.0).explain("text")

    assert recorded["url"] == f"{BASE_URL}/explain"


def test_connection_error_names_the_url_and_how_to_fix_it(monkeypatch):
    _patch_post(monkeypatch, requests.ConnectionError("refused"))

    with pytest.raises(ApiError) as error:
        ApiClient(BASE_URL, 30.0).explain("text")

    message = str(error.value)
    assert BASE_URL in message
    assert "run_api.py" in message


def test_timeout_is_reported_as_a_timeout_not_as_unreachable(monkeypatch):
    _patch_post(monkeypatch, requests.Timeout("too slow"))

    with pytest.raises(ApiError) as error:
        ApiClient(BASE_URL, 30.0).explain("text")

    assert "did not respond" in str(error.value)


def test_422_surfaces_the_field_naming_message_from_the_server(monkeypatch):
    body = {"detail": [{"loc": ["body", "text"], "msg": "text must not be blank."}]}
    _patch_post(monkeypatch, FakeResponse(422, body))

    with pytest.raises(ApiError) as error:
        ApiClient(BASE_URL, 30.0).explain("   ")

    message = str(error.value)
    assert "422" in message
    assert "text must not be blank." in message


def test_503_surfaces_the_string_detail(monkeypatch):
    _patch_post(monkeypatch, FakeResponse(503, {"detail": "Model is not loaded."}))

    with pytest.raises(ApiError) as error:
        ApiClient(BASE_URL, 30.0).explain("text")

    assert "Model is not loaded." in str(error.value)


def test_non_json_body_does_not_raise_a_raw_value_error(monkeypatch):
    _patch_post(
        monkeypatch, FakeResponse(200, FakeResponse._UNDECODABLE, text="<html>")
    )

    with pytest.raises(ApiError) as error:
        ApiClient(BASE_URL, 30.0).explain("text")

    assert "could not read" in str(error.value)


def test_missing_field_in_response_becomes_a_readable_api_error(monkeypatch):
    incomplete = {key: value for key, value in EXPLAIN_BODY.items() if key != "label"}
    _patch_post(monkeypatch, FakeResponse(200, incomplete))

    with pytest.raises(ApiError) as error:
        ApiClient(BASE_URL, 30.0).explain("text")

    assert "could not read" in str(error.value)
