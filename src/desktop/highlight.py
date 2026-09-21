# highlight.py
# Purpose: paint token attributions onto a Tkinter Text widget. Rendering only.
# Author: Colile Sibanda

from __future__ import annotations

import tkinter as tk
from collections.abc import Sequence

from desktop.api_client import TokenWeight

# Five steps per direction is enough to read intensity without implying a
# precision the weights do not have.
_SHADES = 5

PHISHING_TAG_PREFIX = "phishing_"
LEGITIMATE_TAG_PREFIX = "legitimate_"

# Light backgrounds, darkest last, so black body text stays readable on all of
# them. Red pushes towards phishing, blue away from it: blue rather than green
# so the pair stays distinguishable for red-green colour blindness.
_PHISHING_COLOURS = ("#ffe8e8", "#ffd0d0", "#ffb3b3", "#ff9494", "#ff7070")
_LEGITIMATE_COLOURS = ("#e8f0ff", "#d0e0ff", "#b3ceff", "#94b9ff", "#70a3ff")


def legend_colours() -> tuple[str, str]:
    """Return the two swatch colours the app's legend labels.

    The strongest shade of each direction, so the legend matches the most
    saturated highlight a reader will see.

    Inputs: none.
    Output: (phishing colour, legitimate colour) as hex strings.
    """
    return _PHISHING_COLOURS[-1], _LEGITIMATE_COLOURS[-1]


def configure_tags(widget: tk.Text) -> None:
    """Register the background colour tags the renderer assigns.

    Called once when the widget is built, so `render_attributions` only has to
    add ranges.

    Inputs: widget : the Text widget to configure.
    Output: none; mutates the widget's tag table.
    """
    for shade in range(_SHADES):
        widget.tag_configure(
            f"{PHISHING_TAG_PREFIX}{shade}", background=_PHISHING_COLOURS[shade]
        )
        widget.tag_configure(
            f"{LEGITIMATE_TAG_PREFIX}{shade}", background=_LEGITIMATE_COLOURS[shade]
        )


def render_attributions(
    widget: tk.Text, text: str, tokens: Sequence[TokenWeight]
) -> None:
    """Show `text` in the widget with each token shaded by its attribution.

    `text` must be the string the token offsets index into : the normalised
    text the API explained, not the analyst's raw input.
    Highlighting raw text from these offsets would claim attributions the
    model never made.

    Inputs: widget : the Text widget to render into; text : the explained
    string; tokens : token occurrences with signed weights and offsets.
    Output: none; replaces the widget's contents.
    """
    configure_tags(widget)

    widget.configure(state=tk.NORMAL)
    widget.delete("1.0", tk.END)
    widget.insert("1.0", text)

    strongest = max((abs(token.weight) for token in tokens), default=0.0)
    for token in tokens:
        tag = _tag_for(token.weight, strongest)
        if tag is None:
            continue
        widget.tag_add(tag, f"1.0+{token.start}c", f"1.0+{token.end}c")

    widget.configure(state=tk.DISABLED)


def _tag_for(weight: float, strongest: float) -> str | None:
    """Choose the tag for one weight, or None if it should not be shaded.

    Inputs: weight : the token's signed attribution; strongest : the largest
    absolute weight in this explanation, used to scale intensity.
    Output: a tag name, or None when the weight is zero (LIME assigned this
    token nothing) or when no token carried any weight at all.
    """
    if weight == 0.0 or strongest == 0.0:
        return None

    share = abs(weight) / strongest
    shade = min(int(share * _SHADES), _SHADES - 1)
    prefix = PHISHING_TAG_PREFIX if weight > 0 else LEGITIMATE_TAG_PREFIX
    return f"{prefix}{shade}"
