# test_desktop_app.py
# Purpose: unit tests for the window's analyse flow, error handling and guards.
# Author: Colile Sibanda

from __future__ import annotations

import time
import tkinter as tk

import pytest

from desktop import messages
from desktop.api_client import ApiError, ExplainResult, TokenWeight

RESULT = ExplainResult(
    label="phishing",
    probability=0.9731,
    threshold=0.5698,
    model_id="m1_text_cnn",
    text="verify your account now",
    tokens=(
        TokenWeight("verify", 0.21, 0, 6),
        TokenWeight("your", 0.0, 7, 11),
        TokenWeight("account", 0.14, 12, 19),
        TokenWeight("now", -0.04, 20, 23),
    ),
    method="lime",
    caveat="This explanation is local to this prediction.",
)


@pytest.fixture(scope="module")
def _app_instance(tk_root):
    """Build one PhishLensApp for the module, on the session's shared Tk root.

    The app is built into `tk_root` (tests/conftest.py) rather than creating
    its own, because Tk cannot be re-initialised after a root is destroyed in
    the same process. The `app` fixture resets this instance between tests.

    Inputs: tk_root : the session-scoped Tk root.
    Output: a PhishLensApp with its window hidden.
    """
    from desktop.app import PhishLensApp

    return PhishLensApp("http://127.0.0.1:8000", 5.0, root=tk_root)


@pytest.fixture()
def app(_app_instance):
    """Hand each test the shared app, reset to its initial state.

    The app is pointed at a URL that is never contacted: every test replaces
    `client.explain` with a stub, so no request is made.

    Inputs: _app_instance : the module-scoped app.
    Output: the app, cleared before the test runs.
    """
    _app_instance._on_clear()
    return _app_instance


def _pump_until_idle(app, limit: float = 10.0) -> None:
    """Run the Tk event loop until the Analyse button is re-enabled.

    Drives `root.update()` by hand in place of `mainloop()`, so the queue the
    worker posts to is drained exactly as it would be in the running app.

    Inputs: app : the PhishLensApp under test; limit : seconds to wait.
    Output: none. Raises AssertionError if the app never returns to idle,
    which is the symptom of a worker outcome that never reached the GUI.
    """
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        app.root.update()
        if str(app.analyse_button.cget("state")) == "normal":
            return
        time.sleep(0.02)
    raise AssertionError("The app never returned to idle after Analyse.")


def _set_input(app, text: str) -> None:
    """Replace the input area's contents.

    Inputs: app : the PhishLensApp under test; text : the text to enter.
    Output: none.
    """
    app.input_text.delete("1.0", tk.END)
    app.input_text.insert("1.0", text)


def test_blank_input_is_refused_without_calling_the_api(app):
    """Whitespace-only input must never reach the API (src/desktop/README.md)."""
    called = False

    def explain(text):
        nonlocal called
        called = True
        raise AssertionError("the API must not be called for blank input")

    app.client.explain = explain
    _set_input(app, "   \n\t  ")
    app._on_analyse()

    assert called is False
    assert app.status.cget("text") == messages.EMPTY_INPUT
    assert str(app.analyse_button.cget("state")) == "normal"


def test_a_successful_analysis_fills_the_verdict_and_explanation(app):
    app.client.explain = lambda text: RESULT
    _set_input(app, "verify your account now")
    app._on_analyse()
    _pump_until_idle(app)

    verdict = app.verdict_label.cget("text")
    assert "Phishing" in verdict
    assert "97.3%" in verdict  # the probability, never a bare label
    assert "m1_text_cnn" in app.model_label.cget("text")
    assert app.status.cget("text") == messages.STATUS_DONE

    shown = app.explanation_text.get("1.0", tk.END).rstrip("\n")
    assert shown == RESULT.text


def test_an_api_error_shows_a_readable_message_and_recovers(app):
    """The API being down must leave a sentence, not a traceback or a hang.

    This is the regression guard for the worker thread touching Tk directly:
    it raised "main thread is not in main loop", killed the thread, and left
    the button disabled and the status stuck on "Analysing..." forever.
    """
    message = "Cannot reach the PhishLens API at http://127.0.0.1:8000."
    def explain(text):
        raise ApiError(message)

    app.client.explain = explain
    _set_input(app, "verify your account now")
    app._on_analyse()
    _pump_until_idle(app)

    assert app.status.cget("text") == message
    assert str(app.analyse_button.cget("state")) == "normal"


def test_the_worker_thread_never_touches_a_widget(app):
    """The outcome must arrive via the queue, not by the worker rendering it.

    Tk is not thread-safe; a widget write from the worker raises RuntimeError.
    Running the worker directly and asserting the GUI is untouched until the
    queue is drained proves the marshalling is real.
    """
    app.client.explain = lambda text: RESULT
    before = app.verdict_label.cget("text")

    app._analyse_worker("verify your account now")

    assert app.verdict_label.cget("text") == before  # not yet rendered
    assert app._results.qsize() == 1

    app._drain_results()
    assert "Phishing" in app.verdict_label.cget("text")


def test_clear_resets_the_window(app):
    app.client.explain = lambda text: RESULT
    _set_input(app, "verify your account now")
    app._on_analyse()
    _pump_until_idle(app)

    app._on_clear()

    assert app.input_text.get("1.0", tk.END).strip() == ""
    assert app.verdict_label.cget("text") == messages.VERDICT_PLACEHOLDER
    assert app.model_label.cget("text") == ""
    assert app.status.cget("text") == messages.STATUS_READY


def test_the_explanation_panel_is_not_editable_by_the_analyst(app):
    app.client.explain = lambda text: RESULT
    _set_input(app, "verify your account now")
    app._on_analyse()
    _pump_until_idle(app)

    assert str(app.explanation_text.cget("state")) == tk.DISABLED
