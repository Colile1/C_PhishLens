# method_agreement.py
# Purpose: compare a LIME explanation with an Integrated Gradients one for the same email.
# Author: Colile Sibanda

from __future__ import annotations

from dataclasses import dataclass

from phishlens.explain.integrated_gradients import IGExplanation
from phishlens.explain.lime_explainer import Explanation


@dataclass(frozen=True)
class MethodAgreement:
    """How far LIME and IG agree on which tokens drove one M1 prediction.

    Inputs: k : tokens compared per method (LIME's top-list size);
    lime_top, ig_top : each method's top-k word types, strongest first;
    jaccard : overlap of the two sets; shared : types in both;
    sign_agreement : share of shared types both methods push the same way
    (None when nothing is shared).
    Output: immutable MethodAgreement.
    """

    k: int
    lime_top: tuple[str, ...]
    ig_top: tuple[str, ...]
    jaccard: float
    shared: tuple[str, ...]
    sign_agreement: float | None

    def as_dict(self) -> dict[str, object]:
        """Flatten into JSON-writable primitives for the metrics file.

        Inputs: none.
        Output: dict of plain types.
        """
        return {
            "k": self.k,
            "lime_top": list(self.lime_top),
            "ig_top": list(self.ig_top),
            "jaccard": round(self.jaccard, 4),
            "shared": list(self.shared),
            "sign_agreement": None if self.sign_agreement is None else round(self.sign_agreement, 4),
        }


def method_agreement(lime: Explanation, ig: IGExplanation) -> MethodAgreement:
    """Compare the two methods at the word-type level.

    LIME weights word types; IG weights occurrences, so IG is summed per type
    first (`IGExplanation.type_weights`). Both lists rank by absolute weight,
    so evidence for legitimate counts as much as evidence for phishing.

    Inputs: lime : a LIME Explanation of M1; ig : an IGExplanation of M1 for
    the same email.
    Output: a MethodAgreement.

    Raises ValueError if the two explanations are of different texts, which
    would make every comparison meaningless.
    """
    if lime.text != ig.text:
        raise ValueError(
            "LIME and IG explained different normalised texts; compare "
            "explanations of the same email only."
        )

    lime_weights = {entry.token: entry.weight for entry in lime.top}
    lime_top = tuple(entry.token for entry in lime.top)
    k = len(lime_top)

    ig_weights = ig.type_weights()
    ig_top = tuple(sorted(ig_weights, key=lambda token: -abs(ig_weights[token]))[:k])

    union = set(lime_top) | set(ig_top)
    shared = tuple(token for token in lime_top if token in set(ig_top))
    jaccard = 1.0 if not union else len(shared) / len(union)

    sign_agreement = None
    if shared:
        same = sum((lime_weights[t] > 0) == (ig_weights[t] > 0) for t in shared)
        sign_agreement = same / len(shared)

    return MethodAgreement(k, lime_top, ig_top, jaccard, shared, sign_agreement)
