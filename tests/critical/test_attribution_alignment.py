# test_attribution_alignment.py
# Purpose: prove LIME's weights land on the right tokens, at the right offsets, in order.
# Author: Colile Sibanda

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phishlens.config.loader import load_config
from phishlens.explain.attributions import TokenWeight, align_to_tokens, top_tokens
from phishlens.explain.lime_explainer import explain_text
from phishlens.explain.stability import stability_report
from phishlens.features.normalise import normalise_text
from phishlens.features.tokenise import tokenise

# A known string with the three properties that break naive alignment: a
# repeated token ("verify"), a token that is a substring of an earlier one
# ("count" inside "account"), and placeholder tokens carrying angle brackets.
KNOWN_TEXT = "Verify your ACCOUNT at http://evil.example/x within 24 hours or verify count."

PHISHING_LURE = (
    "URGENT: your account has been suspended. Verify your password immediately "
    "at http://secure-update.example/login or your account will be closed "
    "within 24 hours. Click here to confirm your identity now."
)


@pytest.fixture(scope="module")
def config():
    """Load the project config once for the whole module."""
    return load_config(Path("configs/default.yaml"))


def test_alignment_covers_every_token_in_order(config):
    """align_to_tokens emits one entry per token occurrence, in reading order."""
    normalised = normalise_text(KNOWN_TEXT, config)
    expected = tokenise(normalised)

    aligned = align_to_tokens(normalised, {})

    assert [entry.token for entry in aligned] == expected, (
        "Aligned tokens must match tokenise() exactly, in order: any difference "
        "is the misalignment this test exists to catch."
    )


def test_offsets_slice_back_to_the_token(config):
    """Every reported offset pair slices the explained text back to its own token.

    This is the off-by-one guard in its strongest form: it is not enough that
    the tokens are right and the offsets are plausible, the offsets must
    reproduce the token from the very string the app will highlight.
    """
    normalised = normalise_text(KNOWN_TEXT, config)

    for entry in align_to_tokens(normalised, {}):
        assert normalised[entry.start : entry.end] == entry.token, (
            f"Offsets [{entry.start}:{entry.end}] slice "
            f"{normalised[entry.start:entry.end]!r} but the token is {entry.token!r}."
        )


def test_offsets_are_strictly_increasing_and_non_overlapping(config):
    """Successive tokens occupy successive, disjoint spans."""
    normalised = normalise_text(KNOWN_TEXT, config)
    aligned = align_to_tokens(normalised, {})

    for previous, current in zip(aligned, aligned[1:]):
        assert current.start >= previous.end, (
            f"Token {current.token!r} starts at {current.start}, inside the span of "
            f"{previous.token!r} which ends at {previous.end}."
        )


def test_repeated_token_gets_distinct_offsets_and_one_shared_weight(config):
    """A repeated word resolves to successive occurrences, not the first one twice.

    LIME in bag-of-words mode scores the word type once; both occurrences must
    carry that weight while pointing at their own positions in the text.
    """
    normalised = normalise_text(KNOWN_TEXT, config)
    aligned = align_to_tokens(normalised, {"verify": 0.4})

    occurrences = [entry for entry in aligned if entry.token == "verify"]
    assert len(occurrences) == 2, "The fixture text contains 'verify' exactly twice."
    assert occurrences[0].start != occurrences[1].start, (
        "Both occurrences of 'verify' were reported at the same offset: the "
        "forward-only scan is not advancing its cursor."
    )
    assert all(entry.weight == pytest.approx(0.4) for entry in occurrences)


def test_substring_token_is_not_matched_inside_a_longer_token(config):
    """'count' aligns to its own word, not to the 'count' inside 'account'."""
    normalised = normalise_text(KNOWN_TEXT, config)
    aligned = align_to_tokens(normalised, {})

    account = next(entry for entry in aligned if entry.token == "account")
    count = next(entry for entry in aligned if entry.token == "count")

    assert count.start >= account.end, (
        "'count' was matched inside 'account': the scan is finding substrings "
        "rather than the tokeniser's own token boundaries."
    )
    assert normalised[count.start : count.end] == "count"


def test_unscored_tokens_are_reported_with_zero_weight(config):
    """Tokens LIME did not score appear with weight 0.0 rather than being dropped."""
    normalised = normalise_text(KNOWN_TEXT, config)
    aligned = align_to_tokens(normalised, {"verify": 1.0})

    unscored = [entry for entry in aligned if entry.token != "verify"]
    assert unscored, "The fixture has tokens other than 'verify'."
    assert all(entry.weight == 0.0 for entry in unscored)


def test_weights_for_absent_tokens_are_never_emitted(config):
    """A weight map entry with no occurrence in the text produces no TokenWeight."""
    normalised = normalise_text(KNOWN_TEXT, config)
    aligned = align_to_tokens(normalised, {"bitcoin": 9.9})

    assert all(entry.token != "bitcoin" for entry in aligned), (
        "A token absent from the text was emitted: alignment is being driven by "
        "the weight map rather than by the text's own tokens."
    )


def test_placeholder_tokens_survive_alignment(config):
    """The <url> and <num> placeholders align as single tokens with correct spans."""
    normalised = normalise_text(KNOWN_TEXT, config)
    aligned = align_to_tokens(normalised, {})
    tokens = [entry.token for entry in aligned]

    assert "<url>" in tokens and "<num>" in tokens
    for entry in aligned:
        if entry.token in {"<url>", "<num>"}:
            assert normalised[entry.start : entry.end] == entry.token


def test_empty_text_aligns_to_nothing(config):
    """Empty input yields an empty attribution list rather than raising."""
    assert align_to_tokens("", {"verify": 1.0}) == []


def test_top_tokens_ranks_by_absolute_weight(config):
    """Evidence for legitimate ranks alongside evidence for phishing."""
    aligned = [
        TokenWeight("mild", 0.10, 0, 4),
        TokenWeight("strongneg", -0.90, 5, 14),
        TokenWeight("strongpos", 0.50, 15, 24),
    ]

    ranked = top_tokens(aligned, count=2)

    assert [entry.token for entry in ranked] == ["strongneg", "strongpos"], (
        "top_tokens must rank by |weight|, so a strong negative outranks a "
        "weaker positive."
    )


def test_top_tokens_rejects_non_positive_count():
    """A zero or negative count is a caller error and is named as such."""
    with pytest.raises(ValueError, match="count must be positive"):
        top_tokens([], count=0)


# --- End-to-end: the wrapper against a transparent stand-in model ------------


def _keyword_predict_fn(keyword: str):
    """Build a probability function that keys only on one word's presence.

    A stand-in with a known ground truth: because only `keyword` moves the
    output, LIME must attribute to that token and to no other. This tests the
    wrapper's wiring without depending on a trained checkpoint.

    Inputs: keyword : the token the stand-in reacts to.
    Output: a two-column probability function in the project's contract.
    """

    def predict(texts: list[str]) -> np.ndarray:
        phishing = np.array(
            [0.95 if keyword in text.lower() else 0.05 for text in texts],
            dtype=np.float64,
        )
        return np.column_stack([1.0 - phishing, phishing])

    return predict


def test_explanation_attributes_to_the_token_that_drives_the_model(config):
    """LIME finds the one token a transparent stand-in model actually keys on."""
    explanation = explain_text(PHISHING_LURE, _keyword_predict_fn("suspended"), config)

    top = explanation.top
    assert top, "The explanation surfaced no tokens at all."
    assert top[0].token == "suspended", (
        f"The stand-in model keys only on 'suspended', so it must rank first; "
        f"got {[entry.token for entry in top[:3]]}."
    )
    assert top[0].weight > 0.0, "Presence of the driving token must push towards phishing."


def test_explanation_offsets_index_into_its_own_text(config):
    """Explanation.text is the string the offsets are valid against."""
    explanation = explain_text(PHISHING_LURE, _keyword_predict_fn("suspended"), config)

    for entry in explanation.tokens:
        assert explanation.text[entry.start : entry.end] == entry.token


def test_explanation_tokens_match_the_normalised_text_in_order(config):
    """The full token list is the input's own tokens, in order, with no gaps."""
    explanation = explain_text(PHISHING_LURE, _keyword_predict_fn("suspended"), config)

    assert [entry.token for entry in explanation.tokens] == tokenise(
        normalise_text(PHISHING_LURE, config)
    )


def test_explanation_is_reproducible_under_one_seed(config):
    """The same seed twice gives the same weights: no hidden global randomness."""
    predict = _keyword_predict_fn("suspended")

    first = explain_text(PHISHING_LURE, predict, config, seed=42)
    second = explain_text(PHISHING_LURE, predict, config, seed=42)

    assert [entry.token for entry in first.top] == [entry.token for entry in second.top]
    assert [entry.weight for entry in first.top] == pytest.approx(
        [entry.weight for entry in second.top]
    )


def test_blank_text_is_rejected_with_an_actionable_error(config):
    """An email with no tokens names the problem instead of failing inside LIME."""
    with pytest.raises(ValueError, match="no tokens"):
        explain_text("   !!!   ", _keyword_predict_fn("suspended"), config)


def test_stability_report_holds_probability_fixed_across_seeds(config):
    """Only LIME's sampling varies: the model's probability must not move."""
    report = stability_report(
        PHISHING_LURE, _keyword_predict_fn("suspended"), config, seeds=(42, 1337)
    )

    assert report.probability_spread == pytest.approx(0.0, abs=1e-12), (
        "P(phishing) moved between seeds, so something other than LIME's sampling "
        "is non-deterministic in the predict path."
    )
    assert "suspended" in report.agreed_tokens, (
        "The one token driving the stand-in model should surface under both seeds."
    )
    assert 0.0 <= report.jaccard <= 1.0


def test_stability_report_needs_at_least_two_seeds(config):
    """One seed cannot disagree with anything; the error says so."""
    with pytest.raises(ValueError, match="at least two seeds"):
        stability_report(
            PHISHING_LURE, _keyword_predict_fn("suspended"), config, seeds=(42,)
        )
