# render.py
# Purpose: turn an /explain-shaped result into notebook HTML. Rendering only, no model calls.
# Author: Colile Sibanda

from __future__ import annotations

from html import escape

# Five shades per direction: enough to read intensity without implying more
# precision than the weights have. Matches the desktop and browser clients.
SHADES = 5
PHISHING_SHADES = ("#ffe8e8", "#ffd0d0", "#ffb3b3", "#ff9494", "#ff7070")
LEGITIMATE_SHADES = ("#e8f0ff", "#d0e0ff", "#b3ceff", "#94b9ff", "#70a3ff")
CAUTION = (
    "This model assists triage: it does not decide. An analyst confirms every "
    "verdict, and the highlights explain this one prediction only."
)


def render_result(result: dict, top_n: int = 5) -> str:
    """Build the full HTML block for one analysed email.

    Every piece of email text is HTML-escaped, so a pasted email cannot inject
    markup or script into the notebook. Direction is shown three ways (colour,
    bold for phishing and italic for legitimate, and the words in the table),
    so it survives viewers that strip colour.

    Inputs: result : dict in the `/explain` response shape; top_n : signals
    listed per direction.
    Output: an HTML string.
    """
    return "".join([
        _verdict(result),
        "<p><b>Why:</b> words in <b>bold</b> pushed towards phishing, words in "
        "<i>italic</i> towards legitimate; darker shading means a larger weight.</p>",
        f"<div style='font-family:monospace;line-height:1.9;white-space:pre-wrap'>{_highlight(result)}</div>",
        _signals_table(result["tokens"], top_n),
        f"<p><i>{escape(CAUTION)} {escape(result.get('caveat', ''))}</i></p>",
    ])


def _verdict(result: dict) -> str:
    """The one-line answer, stated before any evidence.

    Inputs: result : the `/explain`-shaped dict.
    Output: HTML for the verdict line.
    """
    phishing = result["label"] == "phishing"
    side = "at or above" if phishing else "below"
    headline = "Likely phishing" if phishing else "Likely legitimate"
    return (
        f"<p style='font-size:1.15em'><b>{headline}</b>: P(phishing) "
        f"{_probability(result['probability'])}, {side} the decision threshold of "
        f"{result['threshold']:.4f} (model {escape(result['model_id'])}).</p>"
    )


def _probability(value: float) -> str:
    """Format a probability without rounding it to exactly 0 or 1.

    Inputs: value : P(phishing) in [0, 1].
    Output: e.g. "= 0.9936", "> 0.9999" or "< 0.0001".
    """
    if value > 0.9999:
        return "> 0.9999"
    if value < 0.0001:
        return "< 0.0001"
    return f"= {value:.4f}"


def _highlight(result: dict) -> str:
    """Shade each token in the normalised text by its signed weight.

    The offsets index into `result["text"]`, the normalised string the model
    scored, never into the raw input.

    Inputs: result : the `/explain`-shaped dict.
    Output: HTML with every text segment escaped.
    """
    text, tokens = result["text"], sorted(result["tokens"], key=lambda t: t["start"])
    strongest = max((abs(t["weight"]) for t in tokens), default=0.0)
    parts, cursor = [], 0
    for token in tokens:
        if token["weight"] == 0.0 or strongest == 0.0 or token["start"] < cursor:
            continue
        step = min(int(abs(token["weight"]) / strongest * SHADES), SHADES - 1)
        towards_phishing = token["weight"] > 0
        colour = (PHISHING_SHADES if towards_phishing else LEGITIMATE_SHADES)[step]
        word = escape(text[token["start"]:token["end"]])
        word = f"<b>{word}</b>" if towards_phishing else f"<i>{word}</i>"
        parts.append(escape(text[cursor:token["start"]]))
        parts.append(f"<span style='background:{colour}' title='weight {token['weight']:+.4f}'>{word}</span>")
        cursor = token["end"]
    parts.append(escape(text[cursor:]))
    return "".join(parts)


def top_signals(tokens: list[dict], top_n: int) -> tuple[list[dict], list[dict]]:
    """Rank the tokens that pushed hardest each way.

    A repeated token is listed once, under its largest absolute weight,
    compared case-insensitively. Zero weights are dropped.

    Inputs: tokens : token dicts with "token" and "weight"; top_n : per direction.
    Output: (towards phishing, towards legitimate), strongest first.
    """
    best: dict[str, dict] = {}
    for token in tokens:
        if token["weight"] == 0.0:
            continue
        key = token["token"].lower()
        if key not in best or abs(token["weight"]) > abs(best[key]["weight"]):
            best[key] = token
    ranked = sorted(best.values(), key=lambda t: -abs(t["weight"]))
    return ([t for t in ranked if t["weight"] > 0][:top_n],
            [t for t in ranked if t["weight"] < 0][:top_n])


def _signals_table(tokens: list[dict], top_n: int) -> str:
    """Tabulate the strongest signals each way.

    Inputs: tokens : token dicts; top_n : rows per direction.
    Output: an HTML table, or a sentence when no token carried weight.
    """
    phishing, legitimate = top_signals(tokens, top_n)
    rows = [(t, "towards phishing") for t in phishing] + [(t, "towards legitimate") for t in legitimate]
    if not rows:
        return "<p>No word carried any weight in this explanation.</p>"
    cell = "style='border:1px solid #000;background:#fff;color:#000;padding:4px 10px;text-align:left'"
    body = "".join(
        f"<tr><td {cell}>{escape(t['token'])}</td><td {cell}>{direction}</td><td {cell}>{t['weight']:+.4f}</td></tr>"
        for t, direction in rows
    )
    head = "".join(f"<th {cell}>{name}</th>" for name in ("word", "direction", "weight"))
    # Black and white, with explicit cell styles so notebook themes cannot shade rows.
    return f"<table style='border-collapse:collapse'><tr>{head}</tr>{body}</table>"
