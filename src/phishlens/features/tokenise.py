# tokenise.py
# Purpose: split normalised text into the plain tokens LIME and the encoder both see.
# Author: Colile Sibanda

from __future__ import annotations

import re

# Deliberately plain: word characters and the placeholder angle brackets, so
# "<url>" survives as one token rather than being split into "<", "url", ">".
# Anything else (punctuation, quotes) is a separator, never a token.
_TOKEN = re.compile(r"<[a-z]+>|[a-z0-9']+", re.IGNORECASE)


def tokenise(text: str) -> list[str]:
    """Split normalised text into a list of tokens.

    A simple pattern-based tokeniser, chosen over a library tokeniser so the
    tokens a LIME explanation reports are exactly the tokens the model saw at
    encode time : no hidden subword splitting or stemming in between.

    Inputs: text : normalised string (see normalise_text).
    Output: list of tokens in order; [] for empty or non-textual input.
    """
    if not isinstance(text, str) or not text:
        return []
    return _TOKEN.findall(text)
