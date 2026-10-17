# test_ig_alignment.py
# Purpose: prove Integrated Gradients scores land on the right tokens, with the project's sign.
# Author: Colile Sibanda

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from phishlens.config.loader import load_config
from phishlens.explain.integrated_gradients import integrated_gradients
from phishlens.explain.lime_explainer import explain_text
from phishlens.explain.method_agreement import method_agreement
from phishlens.explain.predict_fn import make_text_cnn_predict_fn
from phishlens.features.vocabulary import PAD_INDEX, UNK_INDEX, Vocabulary
from phishlens.models.text_cnn import TextCNN

# "verify" repeats, "count" hides inside "account", and the URL and number
# become placeholders: the three shapes that break naive alignment.
PLANTED_TEXT = "Verify your ACCOUNT at http://evil.example/x within 24 hours or verify count."
PLANTED = "verify"

PHISHING_LURE = (
    "URGENT: your account has been suspended. Verify your password immediately "
    "at http://secure-update.example/login or your account will be closed "
    "within 24 hours. Click here to confirm your identity now."
)
CHECKPOINT = Path("artifacts/models/text_cnn_best.pt")


@pytest.fixture(scope="module")
def config():
    """Load the project config once for the whole module."""
    return load_config(Path("configs/default.yaml"))


def _planted_model() -> tuple[TextCNN, Vocabulary]:
    """Build a Text-CNN where only two planted words can move the logit.

    Every embedding is zero except "verify" (dimension 0) and "count"
    (dimension 1). Filter 0 reads only dimension 0 and filter 1 only
    dimension 1, both with positive weights; the classifier is positive with
    zero bias. So the baseline logit is exactly 0, both planted words raise
    the logit, and every other token equals the baseline and must receive
    exactly zero attribution.

    Max-over-time pooling credits only the winning window per filter, so of
    the two "verify" occurrences exactly one is credited.

    Inputs: none.
    Output: (model, vocabulary).
    """
    vocabulary = Vocabulary(
        {"<pad>": PAD_INDEX, "<unk>": UNK_INDEX, "verify": 2, "count": 3, "your": 4, "account": 5},
        "<pad>", "<unk>",
    )
    model = TextCNN(vocabulary.size(), 4, (3,), 2, 0.0)
    with torch.no_grad():
        model.embedding.weight.zero_()
        model.embedding.weight[2, 0] = 1.0
        model.embedding.weight[3, 1] = 1.0
        conv = model.convolutions[0]
        conv.weight.zero_()
        conv.weight[0, 0, :] = 0.5
        conv.weight[1, 1, :] = 0.5
        conv.bias.zero_()
        model.classifier.weight.fill_(1.0)
        model.classifier.bias.zero_()
    return model, vocabulary


def test_attribution_lands_only_on_the_planted_tokens(config):
    model, vocabulary = _planted_model()
    result = integrated_gradients(PLANTED_TEXT, model, vocabulary, config)

    for entry in result.tokens:
        assert result.text[entry.start : entry.end] == entry.token
        if entry.token not in (PLANTED, "count"):
            assert entry.weight == 0.0, f"{entry.token!r} equals the baseline yet got {entry.weight}"

    (count,) = [e for e in result.tokens if e.token == "count"]
    assert count.weight > 0.0, "'count' raised the logit, so it must be positive"
    assert count.start == result.text.rindex("count"), "credited 'count' inside 'account'"
    verify_weights = [e.weight for e in result.tokens if e.token == PLANTED]
    assert len(verify_weights) == 2
    assert sum(w > 0.0 for w in verify_weights) == 1, "max pooling credits one winning window"
    assert sum(w == 0.0 for w in verify_weights) == 1


def test_attributions_sum_to_the_logit_gap(config):
    model, vocabulary = _planted_model()
    result = integrated_gradients(PLANTED_TEXT, model, vocabulary, config)

    assert result.baseline_logit == pytest.approx(0.0, abs=1e-6)
    total = sum(entry.weight for entry in result.tokens)
    assert total == pytest.approx(result.logit - result.baseline_logit, rel=0.05, abs=1e-3)


def test_tokens_past_the_sequence_limit_get_zero(config):
    short = replace(config, features=replace(config.features, max_sequence_length=6))
    model, vocabulary = _planted_model()
    result = integrated_gradients(PLANTED_TEXT, model, vocabulary, short)

    assert result.tokens_seen == 6
    first, second = [e for e in result.tokens if e.token == PLANTED]
    assert result.tokens.index(first) < 6 <= result.tokens.index(second)
    assert first.weight > 0.0
    assert second.weight == 0.0, "the model never saw the second 'verify'"


def test_agreement_refuses_different_texts(config):
    model, vocabulary = _planted_model()
    ig = integrated_gradients(PLANTED_TEXT, model, vocabulary, config)
    lime = explain_text("verify your account", make_text_cnn_predict_fn(model, vocabulary, config), config)
    with pytest.raises(ValueError, match="different normalised texts"):
        method_agreement(lime, ig)


@pytest.fixture(scope="module")
def served(config):
    """The committed M1 checkpoint and vocabulary, or skip if absent."""
    if not CHECKPOINT.exists():
        pytest.skip("Served M1 checkpoint not present.")
    from phishlens.serving.loading import load_vocabulary_and_model

    vocabulary, model = load_vocabulary_and_model(config)
    return model, vocabulary


def test_real_lure_attributes_towards_phishing(config, served):
    model, vocabulary = served
    result = integrated_gradients(PHISHING_LURE, model, vocabulary, config)

    assert result.logit > result.baseline_logit
    weights = result.type_weights()
    strongest = sorted(weights, key=lambda t: -abs(weights[t]))[:5]
    assert sum(weights[t] for t in strongest) > 0.0


def test_real_lure_agreement_is_well_formed(config, served):
    model, vocabulary = served
    ig = integrated_gradients(PHISHING_LURE, model, vocabulary, config)
    lime = explain_text(PHISHING_LURE, make_text_cnn_predict_fn(model, vocabulary, config), config)
    agreement = method_agreement(lime, ig)

    assert 0.0 <= agreement.jaccard <= 1.0
    assert len(agreement.ig_top) == agreement.k == len(agreement.lime_top)
