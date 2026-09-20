# lime_explainer.py
# Purpose: wrap LimeTextExplainer so one email yields reproducible token weights.
# Author: Colile Sibanda

from __future__ import annotations

from dataclasses import dataclass

from lime.lime_text import LimeTextExplainer

from phishlens.config.schema import Config
from phishlens.explain.attributions import TokenWeight, align_to_tokens, top_tokens
from phishlens.explain.predict_fn import PHISHING_COLUMN, ProbabilityFn
from phishlens.features.normalise import normalise_text
from phishlens.features.tokenise import tokenise

CLASS_NAMES = ("legitimate", "phishing")


@dataclass(frozen=True)
class Explanation:
    """A local explanation of one prediction.

    Local in the strict sense of Ribeiro et al. (2016): these weights describe
    this one prediction in the neighbourhood of this one email, not the model
    in general. The report and the desktop app both have to say so.

    Inputs: text : the normalised text that was explained, which is also the
    string the offsets index into; probability : P(phishing) for that text;
    tokens : every token occurrence with its weight, in reading order; top :
    the strongest tokens by absolute weight; seed : the seed LIME ran under.
    Output: immutable Explanation.
    """

    text: str
    probability: float
    tokens: list[TokenWeight]
    top: list[TokenWeight]
    seed: int

    def top_token_set(self) -> set[str]:
        """Return the distinct token strings in the top list.

        Used by the stability check, which compares which tokens surfaced
        under different seeds rather than their exact weights.

        Inputs: none.
        Output: set of token strings.
        """
        return {entry.token for entry in self.top}


def explain_text(
    text: str,
    predict_fn: ProbabilityFn,
    config: Config,
    seed: int | None = None,
) -> Explanation:
    """Explain one email's phishing prediction as signed per-token weights.

    LIME is handed the *normalised* text, not the raw email, and the project
    tokeniser is handed to it as `split_expression`. Both
    halves of that matter:

    - Normalised, because LIME's `_segment_with_tokens` requires every token
      the tokeniser returns to appear literally in the string it was given.
      `normalise_text` lowercases and substitutes `<url>`/`<num>`, so tokens
      drawn from normalised text do not occur in the raw text and LIME raises
      "Tokenization produced tokens that do not belong in string!". More
      importantly, the normalised text is what the model actually sees, so it
      is the only string an attribution can honestly point at.
    - The project tokeniser, so the units LIME perturbs are exactly the units
      the encoder turns into vocabulary indices : no hidden re-splitting
      between what is explained and what is scored.

    Positive weights push towards phishing; see attributions.PHISHING_IS_POSITIVE.

    Inputs: text : the email text, raw or normalised (normalisation is
    idempotent); predict_fn : a two-column probability function from
    `predict_fn.py`, the same path the API uses; config : validated Config
    supplying the explain section and the default seed; seed : overrides
    config.seed, used by the stability check.
    Output: an Explanation whose offsets index into `Explanation.text`.

    Raises ValueError naming the problem if the text has no tokens to explain.
    """
    normalised = normalise_text(text, config)
    if not tokenise(normalised):
        raise ValueError(
            "Cannot explain an email with no tokens. After normalisation the "
            "text was empty, so there is nothing for LIME to perturb. Check "
            "that the input is not blank or purely punctuation."
        )

    settings = config.explain
    effective_seed = config.seed if seed is None else seed

    explainer = LimeTextExplainer(
        class_names=list(CLASS_NAMES),
        # The project tokeniser, so LIME perturbs the model's own units.
        split_expression=tokenise,
        # Bag of words: one weight per word type. align_to_tokens expands that
        # single weight back onto every occurrence of the type.
        bow=True,
        random_state=effective_seed,
    )

    lime_explanation = explainer.explain_instance(
        normalised,
        predict_fn,
        labels=(PHISHING_COLUMN,),
        num_features=settings.num_features,
        num_samples=settings.num_samples,
    )

    weights = dict(lime_explanation.as_list(label=PHISHING_COLUMN))
    aligned = align_to_tokens(normalised, weights)

    return Explanation(
        text=normalised,
        probability=float(lime_explanation.predict_proba[PHISHING_COLUMN]),
        tokens=aligned,
        top=top_tokens(aligned, settings.num_features),
        seed=effective_seed,
    )
