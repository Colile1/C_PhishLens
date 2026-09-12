# hashing.py
# Purpose: aggressive content normalisation and hashing for near-duplicate grouping.
# Author: Colile Sibanda

from __future__ import annotations

import hashlib
import re
import unicodedata

# Normalisation used for grouping only. It is deliberately more destructive
# than the model's own preprocessing (features/, the build): two emails
# from the same template that differ only in a name, a date, an amount or a
# tracking id must collapse to one hash, because those are exactly the
# near-duplicates that leak across a random split. This text is never a model
# input.
_DIGITS = re.compile(r"\d+")
_NON_ALPHANUMERIC = re.compile(r"[^a-z\s]+")
_WHITESPACE = re.compile(r"\s+")


def normalise_for_hash(text: str) -> str:
    """Reduce an email to the token skeleton used for duplicate grouping.

    Applies Unicode NFKC folding, lowercases, removes digits and every
    non-alphabetic character, then collapses whitespace. What survives is the
    sequence of alphabetic words, which is stable across the substitutions a
    phishing template makes per victim.

    Inputs: text — the combined subject and body of one email.
    Output: the normalised string; "" for empty or non-textual input.
    """
    if not isinstance(text, str):
        return ""
    folded = unicodedata.normalize("NFKC", text).lower()
    without_digits = _DIGITS.sub(" ", folded)
    alphabetic_only = _NON_ALPHANUMERIC.sub(" ", without_digits)
    return _WHITESPACE.sub(" ", alphabetic_only).strip()


def content_hash(text: str) -> str:
    """Hash an email's normalised content into a stable group id.

    Two emails whose normalised skeletons are identical receive the same id
    and are therefore assigned to the same split as a unit.

    Inputs: text — the combined subject and body of one email.
    Output: a 64-character hexadecimal SHA-256 digest.
    """
    normalised = normalise_for_hash(text)
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()
