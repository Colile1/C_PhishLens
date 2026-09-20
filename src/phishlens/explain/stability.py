# stability.py
# Purpose: measure how far one email's explanation moves when only the seed changes.
# Author: Colile Sibanda

from __future__ import annotations

from dataclasses import dataclass

from phishlens.config.schema import Config
from phishlens.explain.lime_explainer import Explanation, explain_text
from phishlens.explain.predict_fn import ProbabilityFn


@dataclass(frozen=True)
class StabilityReport:
    """Agreement between explanations of one email under different seeds.

    LIME fits a surrogate to a random sample of perturbations, so the top-k
    token set is a random variable. A low overlap does not mean the wrapper is
    broken: it means the explanation is not reproducible at the configured
    `num_samples`, which is a limitation the report must state rather than an
    inconvenience to hide (explain/README.md).

    Inputs: seeds : the seeds compared, in order; explanations : one per seed;
    jaccard : intersection over union of the top-k token sets; agreed_tokens :
    tokens surfacing under every seed; probability_spread : max minus min
    P(phishing) across seeds.
    Output: immutable StabilityReport.
    """

    seeds: tuple[int, ...]
    explanations: tuple[Explanation, ...]
    jaccard: float
    agreed_tokens: tuple[str, ...]
    probability_spread: float

    def as_dict(self) -> dict[str, object]:
        """Flatten the report into JSON-writable primitives.

        Inputs: none.
        Output: dict of plain types, for `evaluation.persistence`-style output
        and for the metric JSON the notebook writes.
        """
        return {
            "seeds": list(self.seeds),
            "jaccard": round(self.jaccard, 4),
            "agreed_tokens": list(self.agreed_tokens),
            "probability_spread": round(self.probability_spread, 6),
            "top_tokens_per_seed": {
                str(explanation.seed): [entry.token for entry in explanation.top]
                for explanation in self.explanations
            },
        }


def stability_report(
    text: str,
    predict_fn: ProbabilityFn,
    config: Config,
    seeds: tuple[int, ...] | None = None,
) -> StabilityReport:
    """Explain one email under each seed and report how far the answers agree.

    The probability spread is the control in this experiment. The model is
    deterministic, so P(phishing) must be identical across seeds : only LIME's
    sampling changed. A non-zero spread would mean something other than the
    seed varied (dropout left on, a non-deterministic predict path), so it is
    reported rather than assumed.

    Inputs: text : the email to explain; predict_fn : two-column probability
    function; config : validated Config; seeds : overrides
    config.explain.stability_seeds.
    Output: a StabilityReport over the requested seeds.

    Raises ValueError if fewer than two seeds are supplied, since a single
    explanation cannot disagree with anything.
    """
    effective_seeds = tuple(seeds if seeds is not None else config.explain.stability_seeds)
    if len(effective_seeds) < 2:
        raise ValueError(
            f"stability_report needs at least two seeds to compare, got "
            f"{effective_seeds!r}. Set explain.stability_seeds in the config."
        )

    explanations = tuple(
        explain_text(text, predict_fn, config, seed=seed) for seed in effective_seeds
    )

    token_sets = [explanation.top_token_set() for explanation in explanations]
    union: set[str] = set().union(*token_sets)
    intersection: set[str] = set(token_sets[0]).intersection(*token_sets[1:])
    # An empty union means no seed surfaced any token; treat that as perfect
    # (vacuous) agreement rather than dividing by zero.
    jaccard = 1.0 if not union else len(intersection) / len(union)

    probabilities = [explanation.probability for explanation in explanations]

    return StabilityReport(
        seeds=effective_seeds,
        explanations=explanations,
        jaccard=jaccard,
        agreed_tokens=tuple(sorted(intersection)),
        probability_spread=max(probabilities) - min(probabilities),
    )
