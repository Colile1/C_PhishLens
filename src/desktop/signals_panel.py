# signals_panel.py
# Purpose: the "strongest signals" lists beside the highlighted text. Rendering only.
# Author: Colile Sibanda

from __future__ import annotations

import tkinter as tk
from collections.abc import Sequence
from tkinter import ttk

from desktop import messages, theme
from desktop.api_client import TokenWeight
from desktop.signals import top_signals

_LIMIT = 6


class SignalsPanel:
    """Two short lists: the tokens that pushed hardest each way.

    Inputs: parent : the widget to build into.
    Output: a SignalsPanel; grid or pack `frame`, then call `show` or `reset`.
    """

    def __init__(self, parent: tk.Misc) -> None:
        self.frame = ttk.Frame(parent, style="Card.TFrame", padding=theme.PADDING)
        self.frame.columnconfigure((0, 1), weight=1, uniform="signals")
        self._columns = []
        for column, (heading, colour) in enumerate((
            (messages.SIGNALS_PHISHING, theme.PHISHING),
            (messages.SIGNALS_LEGITIMATE, theme.LEGITIMATE),
        )):
            tk.Label(self.frame, text=heading, font=theme.FONT_HEADING,
                     background=theme.SURFACE, foreground=colour, anchor="w").grid(
                row=0, column=column, sticky="w", padx=(0, 8))
            body = tk.Label(self.frame, text="", font=theme.FONT_MONO,
                            background=theme.SURFACE, foreground=theme.TEXT,
                            justify="left", anchor="nw")
            body.grid(row=1, column=column, sticky="nw", pady=(4, 0))
            self._columns.append(body)
        self.reset()

    def show(self, tokens: Sequence[TokenWeight]) -> None:
        """List the strongest tokens in each direction.

        Inputs: tokens : the explanation's token occurrences.
        Output: none.
        """
        phishing, legitimate = top_signals(tokens, _LIMIT)
        for body, pairs in zip(self._columns, (phishing, legitimate)):
            body.configure(text=_format_pairs(pairs))

    def reset(self) -> None:
        """Clear both lists back to their placeholder.

        Inputs: none.
        Output: none.
        """
        for body in self._columns:
            body.configure(text=messages.SIGNALS_EMPTY)


def _format_pairs(pairs: list[tuple[str, float]]) -> str:
    """Format (token, weight) pairs as aligned lines.

    Inputs: pairs : tokens with signed weights, strongest first.
    Output: one line per token, or the placeholder when there are none.
    """
    if not pairs:
        return messages.SIGNALS_EMPTY
    width = max(len(token) for token, _ in pairs)
    width = min(width, 18)
    return "\n".join(f"{token[:18]:<{width}}  {weight:+.3f}" for token, weight in pairs)
