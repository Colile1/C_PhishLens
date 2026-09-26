# verdict_card.py
# Purpose: the verdict banner and probability meter. Rendering only.
# Author: Colile Sibanda

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from desktop import messages, theme

_METER_HEIGHT = 14


class VerdictCard:
    """Coloured verdict banner, probability meter with threshold, provenance line.

    Inputs: parent : the widget to build into.
    Output: a VerdictCard; grid or pack `frame`, then call `show` or `reset`.
    Exposes `verdict_label` and `model_label` so the app and its tests can
    read the displayed verdict.
    """

    def __init__(self, parent: tk.Misc) -> None:
        self.frame = ttk.Frame(parent, style="Card.TFrame", padding=0)
        self.frame.columnconfigure(0, weight=1)

        self.banner = tk.Frame(self.frame, background=theme.NEUTRAL)
        self.banner.grid(row=0, column=0, sticky="ew")
        self.headline = tk.Label(
            self.banner, text=messages.HEADLINE_PLACEHOLDER, font=theme.FONT_VERDICT,
            background=theme.NEUTRAL, foreground="#ffffff", anchor="w",
        )
        self.headline.pack(side="left", padx=theme.PADDING, pady=(10, 10))
        self.percent = tk.Label(
            self.banner, text="", font=theme.FONT_VERDICT,
            background=theme.NEUTRAL, foreground="#ffffff",
        )
        self.percent.pack(side="right", padx=theme.PADDING)

        body = ttk.Frame(self.frame, style="Card.TFrame", padding=theme.PADDING)
        body.configure(borderwidth=0)
        body.grid(row=1, column=0, sticky="ew")
        body.columnconfigure(0, weight=1)

        self.verdict_label = ttk.Label(
            body, text=messages.VERDICT_PLACEHOLDER, style="Card.TLabel"
        )
        self.verdict_label.grid(row=0, column=0, sticky="w")
        self.meter = tk.Canvas(
            body, height=_METER_HEIGHT + 12, background=theme.SURFACE,
            highlightthickness=0,
        )
        self.meter.grid(row=1, column=0, sticky="ew", pady=(8, 2))
        self.meter.bind("<Configure>", lambda _event: self._draw_meter())
        self.model_label = ttk.Label(body, text="", style="CardMuted.TLabel")
        self.model_label.grid(row=2, column=0, sticky="w")

        self._probability: float | None = None
        self._threshold: float | None = None

    def show(self, label: str, probability: float, threshold: float,
             model_id: str, method: str) -> None:
        """Render one verdict.

        Inputs: label : "phishing" or "legitimate"; probability : P(phishing);
        threshold : the decision threshold; model_id, method : provenance.
        Output: none.
        """
        colour = theme.PHISHING if label == "phishing" else theme.LEGITIMATE
        self._paint_banner(colour, messages.format_headline(label),
                           f"{probability:.1%}")
        self.verdict_label.configure(
            text=messages.format_verdict(label, probability, threshold)
        )
        self.model_label.configure(text=messages.format_model_line(model_id, method))
        self._probability, self._threshold = probability, threshold
        self._draw_meter()

    def reset(self) -> None:
        """Return the card to its no-verdict state.

        Inputs: none.
        Output: none.
        """
        self._paint_banner(theme.NEUTRAL, messages.HEADLINE_PLACEHOLDER, "")
        self.verdict_label.configure(text=messages.VERDICT_PLACEHOLDER)
        self.model_label.configure(text="")
        self._probability = self._threshold = None
        self._draw_meter()

    def _paint_banner(self, colour: str, headline: str, percent: str) -> None:
        """Recolour the banner and set its two labels.

        Inputs: colour : banner background; headline, percent : label texts.
        Output: none.
        """
        self.banner.configure(background=colour)
        self.headline.configure(text=headline, background=colour)
        self.percent.configure(text=percent, background=colour)

    def _draw_meter(self) -> None:
        """Draw the probability bar with a tick at the decision threshold.

        Inputs: none; reads the last shown probability and threshold.
        Output: none; redraws the canvas.
        """
        canvas = self.meter
        canvas.delete("all")
        width = max(canvas.winfo_width(), 2)
        top, bottom = 6, 6 + _METER_HEIGHT
        canvas.create_rectangle(0, top, width, bottom, fill="#e6e9ef", width=0)
        if self._probability is None or self._threshold is None:
            return

        above = self._probability >= self._threshold
        fill = theme.PHISHING if above else theme.LEGITIMATE
        canvas.create_rectangle(0, top, width * self._probability, bottom,
                                fill=fill, width=0)
        tick = width * self._threshold
        canvas.create_line(tick, 0, tick, bottom + 6, fill=theme.TEXT, width=2)
