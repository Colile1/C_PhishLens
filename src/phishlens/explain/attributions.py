# attributions.py
# Purpose: map LIME's per-word weights back onto tokens in the explained text, in order.
# Author: Colile Sibanda

from __future__ import annotations

from dataclasses import dataclass

from phishlens.features.tokenise import tokenise

# Sign convention, stated once for the whole project (explain/README.md):
# a POSITIVE weight pushes the prediction towards phishing (label 1), a
# NEGATIVE weight pushes it towards legitimate (label 0). Every consumer
# (the notebook, the API, the desktop app) follows this and nothing re-signs.
PHISHING_IS_POSITIVE = True


@dataclass(frozen=True)
class TokenWeight:
    """One token occurrence in the explained text with its attribution weight.

    Position-level, not type-level: a token appearing three times yields three
    TokenWeights sharing one weight, because LIME in bag-of-words mode scores
    a word type, not an occurrence. Keeping them separate is what lets the
    desktop app highlight the exact span the analyst is looking at.

    Inputs: token : the token as it appears in the explained text; weight :
    signed attribution (positive pushes towards phishing); start, end :
    character offsets into the explained (normalised) text, as a Python slice.
    Output: immutable TokenWeight.
    """

    token: str
    weight: float
    start: int
    end: int

    def pushes_towards_phishing(self) -> bool:
        """Report the direction this token pushed the prediction.

        Inputs: none.
        Output: True if the weight is positive (towards phishing), else False.
        """
        return self.weight > 0.0


def align_to_tokens(text: str, weights: dict[str, float]) -> list[TokenWeight]:
    """Attach LIME's per-word weights to every token occurrence in `text`.

    This is the function the build is marked correctness-critical for. It
    walks the text's own tokens in order and looks each one up in the weight
    map, rather than walking the weight map and searching the text. That
    direction is deliberate and is what prevents the off-by-one: the output is
    driven by the input's tokens, so a token can never be reported at another
    token's offsets, and an extra entry in `weights` that does not occur in
    the text is simply never emitted.

    Offsets are found with a forward-only scan (`str.find` from a cursor), so
    repeated tokens resolve to successive occurrences instead of all pointing
    at the first one.

    `text` must be the same string that was handed to LIME : the normalised
    text, not the raw email. Passing raw text here
    would produce offsets into a string the model never saw.

    Inputs: text : the explained text, already normalised; weights : LIME's
    word-to-weight map, typically from `Explanation.as_list()`.
    Output: list of TokenWeight, one per token occurrence in `text`, in the
    order the tokens appear. Tokens LIME did not score are included with a
    weight of 0.0, so the caller can render the full text without holes.
    """
    if not isinstance(text, str) or not text:
        return []

    aligned: list[TokenWeight] = []
    cursor = 0

    for token in tokenise(text):
        start = text.find(token, cursor)
        if start < 0:
            # tokenise() draws its tokens out of this very string, so every
            # token is present at or after the cursor. Reaching this means the
            # tokeniser and this scan disagree, which is exactly the silent
            # misalignment this module exists to prevent: fail loudly.
            raise ValueError(
                f"Token {token!r} was produced by tokenise() but not found in the "
                f"text at or after offset {cursor}. The tokeniser and the offset "
                f"scan disagree; attributions cannot be aligned safely."
            )

        end = start + len(token)
        aligned.append(
            TokenWeight(
                token=token,
                weight=float(weights.get(token, 0.0)),
                start=start,
                end=end,
            )
        )
        cursor = end

    return aligned


def top_tokens(aligned: list[TokenWeight], count: int) -> list[TokenWeight]:
    """Return the `count` most influential token occurrences by absolute weight.

    Absolute value, so the strongest evidence for *legitimate* ranks alongside
    the strongest evidence for phishing. A report that showed only positive
    weights would misrepresent a borderline email as one-sided.

    Duplicate occurrences of one token are de-duplicated to the first
    occurrence, so a word repeated ten times cannot fill the whole list.

    Inputs: aligned : output of align_to_tokens; count : how many to return.
    Output: list of at most `count` TokenWeight, strongest first. Ties are
    broken by position so the order is deterministic.

    Raises ValueError if count is not positive.
    """
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}.")

    seen: set[str] = set()
    unique: list[TokenWeight] = []
    for entry in aligned:
        if entry.token in seen or entry.weight == 0.0:
            continue
        seen.add(entry.token)
        unique.append(entry)

    unique.sort(key=lambda entry: (-abs(entry.weight), entry.start))
    return unique[:count]
