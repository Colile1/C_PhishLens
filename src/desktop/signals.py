# signals.py
# Purpose: summarise token attributions into the strongest signals each way.
# Author: Colile Sibanda

from __future__ import annotations

from collections.abc import Sequence

from desktop.api_client import TokenWeight


def top_signals(
    tokens: Sequence[TokenWeight], limit: int = 5
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Pick the tokens that pushed hardest towards each class.

    A token that occurs several times is listed once, under the occurrence
    with the largest absolute weight, compared case-insensitively. Zero
    weights are dropped: LIME assigned those tokens nothing.

    Inputs: tokens : token occurrences with signed weights, positive towards
    phishing; limit : how many tokens to keep per direction.
    Output: (towards phishing, towards legitimate), each a list of
    (token, weight) sorted strongest first.
    """
    strongest: dict[str, tuple[str, float]] = {}
    for token in tokens:
        if token.weight == 0.0:
            continue
        key = token.token.lower()
        held = strongest.get(key)
        if held is None or abs(token.weight) > abs(held[1]):
            strongest[key] = (token.token, token.weight)

    ranked = sorted(strongest.values(), key=lambda pair: abs(pair[1]), reverse=True)
    phishing = [pair for pair in ranked if pair[1] > 0][:limit]
    legitimate = [pair for pair in ranked if pair[1] < 0][:limit]
    return phishing, legitimate
