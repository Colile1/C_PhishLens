# messages.py
# Purpose: every user-facing string the desktop app shows, in one place.
# Author: Colile Sibanda

from __future__ import annotations

WINDOW_TITLE = "PhishLens : phishing email triage"
APP_TITLE = "PhishLens"
APP_SUBTITLE = "Phishing email triage with token-level explanations"

INPUT_HEADING = "Email text (subject and body)"
INPUT_HINT = "Paste an email or open a .eml / .txt file. Headers are not used."
EXPLANATION_HEADING = "Why the model decided this"
SIGNALS_HEADING = "Strongest signals"
ANALYSE_BUTTON = "Analyse  (Ctrl+Enter)"
CLEAR_BUTTON = "Clear"
PASTE_BUTTON = "Paste"
OPEN_BUTTON = "Open file..."
OPEN_DIALOG_TITLE = "Open an email file"

STATUS_READY = "Ready. Paste an email and click Analyse."
STATUS_ANALYSING = "Analysing: the explanation runs LIME, so this takes a moment."
STATUS_DONE = "Analysis complete."

VERDICT_PLACEHOLDER = "No email analysed yet."
HEADLINE_PLACEHOLDER = "Awaiting analysis"
EXPLANATION_PLACEHOLDER = (
    "Token highlights appear here after an email is analysed."
)

LEGEND_PHISHING = "pushed towards phishing"
LEGEND_LEGITIMATE = "pushed towards legitimate"

SIGNALS_PHISHING = "Towards phishing"
SIGNALS_LEGITIMATE = "Towards legitimate"
SIGNALS_EMPTY = "none yet"

# API status indicator in the header.
API_CHECKING = "API: checking..."
API_ONLINE = "API online"
API_LOADING = "API up : model still loading"
API_OFFLINE = "API offline : run python run_api.py"

# The report makes this same claim, so the app has to match it.
CAUTION = (
    "This model assists triage: it does not decide. An analyst confirms every "
    "verdict, and the highlights explain this one prediction only."
)

EMPTY_INPUT = "Enter some email text before clicking Analyse."
CLIPBOARD_EMPTY = "The clipboard holds no text to paste."
FILE_LOADED = "Loaded {name}. Click Analyse when ready."
FILE_UNREADABLE = "Could not read {name}: {detail}"


def format_word_count(text: str) -> str:
    """Render the input's size under the text area.

    Inputs: text : the current input.
    Output: e.g. "132 words : 811 characters".
    """
    stripped = text.strip()
    words = len(stripped.split()) if stripped else 0
    return f"{words} words : {len(stripped)} characters"


def format_headline(label: str) -> str:
    """Render the banner headline for a verdict.

    Inputs: label : "phishing" or "legitimate".
    Output: the banner text.
    """
    return "Likely phishing" if label == "phishing" else "Likely legitimate"

# Errors. Each names what failed and what to do about it, never a traceback.
API_UNREACHABLE = (
    "Cannot reach the PhishLens API at {url}. Start it with "
    "`python run_api.py` and try again."
)
API_TIMEOUT = (
    "The API at {url} did not respond within {timeout:.0f} seconds. It may "
    "still be loading the model, or the email may be very long."
)
API_ERROR = "The API rejected this request ({status}): {detail}"
API_BAD_RESPONSE = (
    "The API returned a response this app could not read: {detail}"
)


def format_verdict(label: str, probability: float, threshold: float) -> str:
    """Render the verdict line: never a bare label, always the probability too.

    Inputs: label : "phishing" or "legitimate" as returned by the API;
    probability : P(phishing) in [0, 1]; threshold : the decision threshold
    the label was derived from, shown so the operating point is visible.
    Output: a single display string.
    """
    return (
        f"{label.capitalize()} : {probability:.1%} probability of phishing "
        f"(decision threshold {threshold:.1%})"
    )


def format_model_line(model_id: str, method: str) -> str:
    """Render the provenance line, so a demo screenshot traces to a model.

    Inputs: model_id : the served model's id; method : the XAI method name.
    Output: a single display string.
    """
    return f"Model: {model_id}    Explanation method: {method}"
