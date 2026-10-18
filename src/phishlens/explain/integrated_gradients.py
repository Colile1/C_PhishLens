# integrated_gradients.py
# Purpose: Captum Integrated Gradients for M1, aligned to the same token offsets as LIME.
# Author: Colile Sibanda

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from captum.attr import LayerIntegratedGradients
from torch import nn

from phishlens.config.schema import Config
from phishlens.explain.attributions import TokenWeight, align_to_tokens
from phishlens.features.encoder import encode
from phishlens.features.normalise import normalise_text
from phishlens.features.vocabulary import PAD_INDEX, Vocabulary

# Completeness tolerance: |sum(attributions) - (logit(x) - logit(baseline))|
# must stay within this share of the logit gap (or of 1.0 when the gap is
# small). Beyond it the path integral is too coarse to trust; raise ig_steps.
COMPLETENESS_TOLERANCE = 0.05


@dataclass(frozen=True)
class IGExplanation:
    """Integrated Gradients attributions for one email under M1.

    Inputs: text : the normalised text every offset indexes into (as LIME,
    the same convention LIME uses); tokens : one TokenWeight per token occurrence, in
    order, positive towards phishing; logit, baseline_logit : model output for
    the email and for the all-pad baseline; convergence_delta : Captum's
    completeness error; tokens_seen : how many leading tokens fit in
    max_sequence_length (later tokens carry 0.0: the model never saw them).
    Output: immutable IGExplanation.
    """

    text: str
    tokens: tuple[TokenWeight, ...]
    logit: float
    baseline_logit: float
    convergence_delta: float
    tokens_seen: int

    def type_weights(self) -> dict[str, float]:
        """Sum occurrence weights per word type, comparable with LIME's.

        LIME scores a word type by removing every occurrence at once, so the
        matching IG quantity is the sum over that word's occurrences.

        Inputs: none.
        Output: {token: summed weight}, zero-weight types omitted.
        """
        summed: dict[str, float] = {}
        for entry in self.tokens:
            summed[entry.token] = summed.get(entry.token, 0.0) + entry.weight
        return {token: weight for token, weight in summed.items() if weight != 0.0}


def integrated_gradients(
    text: str, model: nn.Module, vocabulary: Vocabulary, config: Config
) -> IGExplanation:
    """Attribute M1's phishing logit to each token occurrence with IG.

    Attributions are taken at the embedding layer against the all-`<pad>`
    baseline, whose embedding is fixed at zero. Encoder
    position i is token i of the normalised text, which is exactly the list
    `align_to_tokens` walks, so scores are attached by position.

    Inputs: text : raw or normalised email; model : TextCNN with best weights;
    vocabulary : its training vocabulary; config : supplies explain.ig_steps
    and features.max_sequence_length.
    Output: an IGExplanation.

    Raises ValueError if the text has no tokens, if padded positions receive
    attribution, or if completeness fails beyond COMPLETENESS_TOLERANCE.
    """
    normalised = normalise_text(text, config)
    aligned = align_to_tokens(normalised, {})
    if not aligned:
        raise ValueError("Cannot attribute an email with no tokens after normalisation.")

    model.eval()
    device = next(model.parameters()).device
    ids = torch.from_numpy(encode([normalised], vocabulary, config)).to(device)
    baseline = torch.full_like(ids, PAD_INDEX)

    explainer = LayerIntegratedGradients(model, model.embedding)
    attributions, delta = explainer.attribute(
        ids, baselines=baseline, n_steps=config.explain.ig_steps,
        return_convergence_delta=True,
    )
    per_position = attributions.sum(dim=-1).squeeze(0).detach().cpu()

    with torch.no_grad():
        logit = float(model(ids)[0])
        baseline_logit = float(model(baseline)[0])

    seen = min(len(aligned), config.features.max_sequence_length)
    _check_padding(per_position, seen)
    _check_completeness(float(per_position.sum()), logit - baseline_logit, config)

    tokens = tuple(
        replace(entry, weight=float(per_position[i]) if i < seen else 0.0)
        for i, entry in enumerate(aligned)
    )
    return IGExplanation(normalised, tokens, logit, baseline_logit, float(delta[0]), seen)


def _check_padding(per_position: torch.Tensor, seen: int) -> None:
    """Refuse attributions on padded positions, where input equals baseline.

    Inputs: per_position : one score per encoder position; seen : real tokens.
    Output: none. Raises ValueError if any padded position is non-zero.
    """
    leaked = per_position[seen:].abs().max().item() if seen < len(per_position) else 0.0
    if leaked > 1e-6:
        raise ValueError(
            f"Padded positions received attribution {leaked:.3g}; positions and "
            f"tokens are misaligned. Check encode() against tokenise()."
        )


def _check_completeness(total: float, gap: float, config: Config) -> None:
    """Refuse an IG run whose attributions do not sum to the logit gap.

    Inputs: total : sum of attributions; gap : logit(x) - logit(baseline);
    config : names the step count in the error.
    Output: none. Raises ValueError when the error exceeds the tolerance.
    """
    error = abs(total - gap)
    if error > COMPLETENESS_TOLERANCE * max(1.0, abs(gap)):
        raise ValueError(
            f"Integrated Gradients completeness failed: attributions sum to "
            f"{total:.4f} but the logit gap is {gap:.4f}. Raise explain.ig_steps "
            f"(currently {config.explain.ig_steps})."
        )
