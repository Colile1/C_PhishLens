# test_notebook_ui.py
# Purpose: unit tests for the notebook client: safe rendering, signal ranking and both backends.
# Author: Colile Sibanda

from __future__ import annotations

from pathlib import Path

import pytest

from notebook_ui import ApiBackend, LocalBackend, render_result, top_signals

PHISHING_LURE = (
    "URGENT: your account has been suspended. Verify your password immediately "
    "at http://secure-update.example/login or your account will be closed "
    "within 24 hours. Click here to confirm your identity now."
)
CHECKPOINT = Path("artifacts/models/text_cnn_best.pt")


def _result(text: str, tokens: list[dict], label: str = "phishing") -> dict:
    """Build a minimal /explain-shaped result for rendering tests."""
    return {"label": label, "probability": 0.91, "threshold": 0.57, "model_id": "m1_text_cnn",
            "text": text, "tokens": tokens, "method": "lime", "caveat": "Local to this prediction."}


def test_email_text_is_escaped_not_rendered_as_markup() -> None:
    text = "<script>alert(1)</script> verify"
    html = render_result(_result(text, [{"token": "verify", "weight": 0.5, "start": 26, "end": 32}]))
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_direction_is_carried_by_bold_and_italic_not_only_colour() -> None:
    text = "verify thanks"
    tokens = [{"token": "verify", "weight": 0.5, "start": 0, "end": 6},
              {"token": "thanks", "weight": -0.4, "start": 7, "end": 13}]
    html = render_result(_result(text, tokens))
    assert "<b>verify</b>" in html
    assert "<i>thanks</i>" in html
    assert "towards phishing" in html and "towards legitimate" in html


def test_top_signals_dedupes_case_insensitively_and_drops_zero() -> None:
    tokens = [{"token": "Verify", "weight": 0.2}, {"token": "verify", "weight": 0.6},
              {"token": "the", "weight": 0.0}, {"token": "thanks", "weight": -0.3}]
    phishing, legitimate = top_signals(tokens, top_n=5)
    assert [t["weight"] for t in phishing] == [0.6]
    assert [t["token"] for t in legitimate] == ["thanks"]


def test_api_backend_names_the_url_when_unreachable() -> None:
    backend = ApiBackend("http://127.0.0.1:9", timeout_seconds=2)
    with pytest.raises(RuntimeError, match="127.0.0.1:9"):
        backend.analyse("hello there")


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Served M1 checkpoint not present.")
def test_local_backend_flags_a_known_lure_with_aligned_offsets() -> None:
    result = LocalBackend().analyse(PHISHING_LURE)
    assert result["label"] == "phishing"
    assert all(result["text"][t["start"]:t["end"]] == t["token"] for t in result["tokens"])
    assert "Likely phishing" in render_result(result)


class _StubBackend:
    """Return a fixed result, or raise, without loading a model."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    def analyse(self, text: str) -> dict:
        if self.error:
            raise self.error
        return _result("verify now", [{"token": "verify", "weight": 0.5, "start": 0, "end": 6}])


def _panel_parts(panel):
    """Pull the text box, Analyse button and output widget out of the panel."""
    box, buttons, output = panel.children
    return box, buttons.children[0], output


def test_panel_renders_a_result_when_analyse_is_clicked() -> None:
    from notebook_ui import launch

    box, analyse_button, output = _panel_parts(launch(_StubBackend()))
    box.value = "verify now"
    analyse_button.click()
    assert "Likely phishing" in output.value
    assert analyse_button.disabled is False


def test_panel_refuses_blank_text_and_shows_backend_errors_escaped() -> None:
    from notebook_ui import launch

    box, analyse_button, output = _panel_parts(launch(_StubBackend()))
    box.value = "   "
    analyse_button.click()
    assert "Enter some email text" in output.value

    box, analyse_button, output = _panel_parts(launch(_StubBackend(RuntimeError("<b>down</b>"))))
    box.value = "hello"
    analyse_button.click()
    assert "&lt;b&gt;down&lt;/b&gt;" in output.value
