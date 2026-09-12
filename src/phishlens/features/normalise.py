# normalise.py
# Purpose: reduce cleaned email text to the surface form both models train on.
# Author: Colile Sibanda

from __future__ import annotations

import re

from phishlens.config.schema import Config

# Order matters: URLs and emails are matched before the generic digit run, so
# a URL containing digits (a port, a query id) is replaced whole rather than
# leaving digit fragments behind for the number placeholder to also match.
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_LONG_DIGIT_RUN = re.compile(r"\d{2,}")
_WHITESPACE = re.compile(r"\s+")


def normalise_text(text: str, config: Config) -> str:
    """Lowercase and replace URLs, email addresses and numbers with placeholders.

    Applied identically ahead of both the TF-IDF vectoriser (M0) and the
    vocabulary/encoder (M1), so the two models see the same surface form and
    LIME's reported tokens match what the model actually saw. Idempotent:
    normalising an already-normalised string returns it unchanged, since the
    placeholder tokens themselves contain no URL, email or multi-digit
    pattern for a second pass to match.

    Inputs: text — cleaned subject+body string; config — validated Config
    supplying the preprocessing switches and placeholder tokens.
    Output: the normalised string; "" for empty or non-textual input.
    """
    if not isinstance(text, str) or not text:
        return ""

    settings = config.preprocess
    result = text.lower() if settings.lowercase else text
    result = _URL.sub(settings.url_placeholder, result)
    result = _EMAIL.sub(settings.email_placeholder, result)
    result = _LONG_DIGIT_RUN.sub(settings.number_placeholder, result)
    return _WHITESPACE.sub(" ", result).strip()
