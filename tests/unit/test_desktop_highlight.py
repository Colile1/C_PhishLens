# test_desktop_highlight.py
# Purpose: unit tests for token highlighting in the desktop explanation panel.
# Author: Colile Sibanda

from __future__ import annotations

import tkinter as tk

import pytest

from desktop import highlight
from desktop.api_client import TokenWeight

# The fixture text is adversarial on purpose, mirroring
# tests/critical/test_attribution_alignment.py: a repeated token ("verify"),
# a token that is a substring of an earlier one ("count" inside "account"),
# and a placeholder the normaliser produces ("<url>").
TEXT = "verify your account at <url> and verify the count now"
TOKENS = (
    TokenWeight(token="verify", weight=0.20, start=0, end=6),
    TokenWeight(token="your", weight=0.0, start=7, end=11),
    TokenWeight(token="account", weight=0.10, start=12, end=19),
    TokenWeight(token="at", weight=-0.05, start=20, end=22),
    TokenWeight(token="<url>", weight=0.40, start=23, end=28),
    TokenWeight(token="and", weight=0.0, start=29, end=32),
    TokenWeight(token="verify", weight=0.20, start=33, end=39),
    TokenWeight(token="the", weight=0.0, start=40, end=43),
    TokenWeight(token="count", weight=-0.15, start=44, end=49),
    TokenWeight(token="now", weight=0.0, start=50, end=53),
)


@pytest.fixture()
def widget(tk_root):
    """Build a fresh Text widget for one test, on the shared session root.

    Tk widgets accept insert and tag operations without `mainloop()`, so the
    renderer is testable without showing a window. The root comes from
    tests/conftest.py, because Tk cannot be re-initialised once a root has
    been destroyed in the same process.

    Inputs: tk_root : the session-scoped Tk root.
    Output: a tk.Text widget, destroyed after the test.
    """
    text_widget = tk.Text(tk_root)
    yield text_widget
    text_widget.destroy()


def _tagged_ranges(widget: tk.Text, tag: str) -> list[tuple[str, str]]:
    """Return the (start, end) index pairs a tag covers.

    Inputs: widget : the rendered Text widget; tag : the tag name to query.
    Output: a list of index pairs as strings, empty when the tag is unused.
    """
    ranges = widget.tag_ranges(tag)
    return [
        (str(ranges[index]), str(ranges[index + 1]))
        for index in range(0, len(ranges), 2)
    ]


def _offset_of(widget: tk.Text, index: str) -> int:
    """Convert a Tk text index to a character offset from the start.

    `Text.count` returns None rather than (0,) for a zero-length span, so
    index "1.0" has to be handled explicitly.

    Inputs: widget : the rendered Text widget; index : a Tk index string.
    Output: the offset in characters from "1.0".
    """
    counted = widget.count("1.0", index, "chars")
    return 0 if counted is None else int(counted[0])


def _all_tagged_spans(widget: tk.Text) -> list[tuple[int, int]]:
    """Return every highlighted span as character offsets, in order.

    Inputs: widget : the rendered Text widget.
    Output: a sorted list of (start, end) character offsets.
    """
    spans: list[tuple[int, int]] = []
    for tag in widget.tag_names():
        if tag == "sel":
            continue
        for start, end in _tagged_ranges(widget, tag):
            spans.append((_offset_of(widget, start), _offset_of(widget, end)))
    return sorted(spans)


def test_the_widget_shows_exactly_the_explained_text(widget):
    highlight.render_attributions(widget, TEXT, TOKENS)

    # Tk appends a trailing newline to a Text widget's contents.
    assert widget.get("1.0", tk.END).rstrip("\n") == TEXT


def test_every_tagged_span_slices_back_to_its_own_token(widget):
    """The off-by-one guard: a span must cover its token and nothing else.

    A misaligned map would highlight a confident, false explanation in the
    demo, which is exactly why the alignment is treated as correctness-critical.
    """
    highlight.render_attributions(widget, TEXT, TOKENS)

    weighted = {
        (token.start, token.end): token.token for token in TOKENS if token.weight != 0.0
    }
    spans = _all_tagged_spans(widget)

    assert len(spans) == len(weighted)
    for start, end in spans:
        assert (start, end) in weighted
        assert TEXT[start:end] == weighted[(start, end)]


def test_zero_weight_tokens_are_not_highlighted(widget):
    highlight.render_attributions(widget, TEXT, TOKENS)

    spans = _all_tagged_spans(widget)
    for token in TOKENS:
        if token.weight == 0.0:
            assert (token.start, token.end) not in spans


def test_a_repeated_token_is_highlighted_at_both_of_its_occurrences(widget):
    highlight.render_attributions(widget, TEXT, TOKENS)

    spans = _all_tagged_spans(widget)
    assert (0, 6) in spans
    assert (33, 39) in spans


def test_sign_decides_the_colour_direction(widget):
    highlight.render_attributions(widget, TEXT, TOKENS)

    phishing_spans: list[tuple[int, int]] = []
    legitimate_spans: list[tuple[int, int]] = []
    for tag in widget.tag_names():
        target = None
        if tag.startswith(highlight.PHISHING_TAG_PREFIX):
            target = phishing_spans
        elif tag.startswith(highlight.LEGITIMATE_TAG_PREFIX):
            target = legitimate_spans
        if target is None:
            continue
        for start, _ in _tagged_ranges(widget, tag):
            target.append(_offset_of(widget, start))

    # "<url>" (+0.40) and "account" (+0.10) push towards phishing;
    # "at" (-0.05) and "count" (-0.15) push away from it.
    assert 23 in phishing_spans
    assert 12 in phishing_spans
    assert 20 in legitimate_spans
    assert 44 in legitimate_spans


def test_the_strongest_token_gets_the_most_saturated_shade(widget):
    highlight.render_attributions(widget, TEXT, TOKENS)

    strongest_tag = f"{highlight.PHISHING_TAG_PREFIX}4"
    starts = [
        _offset_of(widget, start)
        for start, _ in _tagged_ranges(widget, strongest_tag)
    ]
    assert starts == [23]  # "<url>", the largest absolute weight at 0.40


def test_rendering_twice_does_not_leave_stale_highlights(widget):
    highlight.render_attributions(widget, TEXT, TOKENS)
    highlight.render_attributions(
        widget, "all clear now", (TokenWeight("clear", -0.3, 4, 9),)
    )

    assert widget.get("1.0", tk.END).rstrip("\n") == "all clear now"
    assert _all_tagged_spans(widget) == [(4, 9)]


def test_an_explanation_with_no_weights_renders_text_without_tags(widget):
    highlight.render_attributions(
        widget, TEXT, (TokenWeight("verify", 0.0, 0, 6),)
    )

    assert widget.get("1.0", tk.END).rstrip("\n") == TEXT
    assert _all_tagged_spans(widget) == []


def test_the_panel_is_left_read_only(widget):
    highlight.render_attributions(widget, TEXT, TOKENS)

    assert str(widget.cget("state")) == tk.DISABLED
