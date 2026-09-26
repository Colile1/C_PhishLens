# test_desktop_helpers.py
# Purpose: unit tests for the desktop app's signal summary, email file reader and health check.
# Author: Colile Sibanda

from __future__ import annotations

import pytest
import requests

from desktop import api_client, messages
from desktop.api_client import ApiClient, ApiError, TokenWeight
from desktop.email_file import read_email_text
from desktop.signals import top_signals


def test_top_signals_splits_by_sign_and_ranks_by_strength():
    tokens = [
        TokenWeight("verify", 0.21, 0, 6),
        TokenWeight("meeting", -0.30, 7, 14),
        TokenWeight("account", 0.14, 15, 22),
        TokenWeight("the", 0.0, 23, 26),
        TokenWeight("thanks", -0.05, 27, 33),
    ]
    phishing, legitimate = top_signals(tokens, limit=5)

    assert phishing == [("verify", 0.21), ("account", 0.14)]
    assert legitimate == [("meeting", -0.30), ("thanks", -0.05)]


def test_top_signals_lists_a_repeated_token_once_at_its_strongest():
    tokens = [
        TokenWeight("Verify", 0.10, 0, 6),
        TokenWeight("verify", 0.25, 10, 16),
    ]
    phishing, _ = top_signals(tokens)

    assert phishing == [("verify", 0.25)]


def test_top_signals_respects_the_limit():
    tokens = [TokenWeight(f"t{i}", 0.1 * (i + 1), i, i + 1) for i in range(8)]
    phishing, legitimate = top_signals(tokens, limit=3)

    assert [token for token, _ in phishing] == ["t7", "t6", "t5"]
    assert legitimate == []


def test_eml_is_reduced_to_subject_and_body_without_headers(tmp_path):
    """Headers are the corpus artefact the project guards against: never sent."""
    eml = tmp_path / "sample.eml"
    eml.write_bytes(
        b"From: attacker@evil.example\r\n"
        b"To: victim@example.com\r\n"
        b"Subject: Verify your account\r\n"
        b"Content-Type: text/plain; charset=utf-8\r\n\r\n"
        b"Click the link to keep access.\r\n"
    )
    text = read_email_text(eml)

    assert text == "Verify your account\n\nClick the link to keep access."
    assert "evil.example" not in text


def test_txt_is_read_verbatim(tmp_path):
    txt = tmp_path / "sample.txt"
    txt.write_text("Subject line\nBody text", encoding="utf-8")

    assert read_email_text(txt) == "Subject line\nBody text"


def test_an_empty_eml_names_the_file(tmp_path):
    eml = tmp_path / "empty.eml"
    eml.write_bytes(b"From: someone@example.com\r\n\r\n")

    with pytest.raises(ValueError, match="empty.eml"):
        read_email_text(eml)


class _Response:
    """Minimal stand-in for requests.Response in the health tests."""

    def __init__(self, body: dict) -> None:
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._body


def test_health_reports_whether_the_model_is_loaded(monkeypatch):
    monkeypatch.setattr(api_client.requests, "get",
                        lambda url, timeout: _Response({"model_loaded": True}))

    assert ApiClient("http://127.0.0.1:8000", 5.0).health() is True


def test_health_raises_a_readable_error_when_unreachable(monkeypatch):
    def refuse(url, timeout):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(api_client.requests, "get", refuse)

    with pytest.raises(ApiError, match="Cannot reach the PhishLens API"):
        ApiClient("http://127.0.0.1:8000", 5.0).health()


def test_word_count_formats_words_and_characters():
    assert messages.format_word_count("  verify your account \n") == (
        "3 words : 19 characters"
    )
