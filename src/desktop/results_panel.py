# results_panel.py
# Purpose: the right-hand pane: verdict, caution, highlighted text and signals. Rendering only.
# Author: Colile Sibanda

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from desktop import highlight, messages, theme
from desktop.api_client import ExplainResult
from desktop.signals_panel import SignalsPanel
from desktop.verdict_card import VerdictCard


class ResultsPanel:
    """Everything the analyst reads after an analysis, stacked in one pane.

    Inputs: parent : the widget to build into.
    Output: a ResultsPanel exposing `frame`, `verdict_label`, `model_label`
    and `explanation_text`; call `show` or `reset`.
    """

    def __init__(self, parent: tk.Misc) -> None:
        self.frame = ttk.Frame(parent)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(2, weight=1)

        self.verdict_card = VerdictCard(self.frame)
        self.verdict_card.frame.grid(row=0, column=0, sticky="ew", padx=(8, 0))
        self.verdict_label = self.verdict_card.verdict_label
        self.model_label = self.verdict_card.model_label

        caution = ttk.Label(self.frame, text=messages.CAUTION, wraplength=620,
                            foreground=theme.MUTED, font=theme.FONT_SMALL)
        caution.grid(row=1, column=0, sticky="ew", padx=(8, 0), pady=6)
        caution.bind("<Configure>",
                     lambda event: caution.configure(wraplength=max(event.width, 200)))

        self._build_explanation(row=2)
        self.signals_panel = SignalsPanel(self.frame)
        self.signals_panel.frame.grid(row=3, column=0, sticky="ew", padx=(8, 0),
                                      pady=(8, 0))

    def show(self, result: ExplainResult) -> None:
        """Render one analysis into every part of the pane.

        Inputs: result : the parsed /explain response.
        Output: none.
        """
        self.verdict_card.show(result.label, result.probability, result.threshold,
                               result.model_id, result.method)
        highlight.render_attributions(self.explanation_text, result.text, result.tokens)
        self.signals_panel.show(result.tokens)

    def reset(self) -> None:
        """Return every part of the pane to its empty state.

        Inputs: none.
        Output: none.
        """
        self.verdict_card.reset()
        self.signals_panel.reset()
        self.explanation_text.configure(state=tk.NORMAL)
        self.explanation_text.delete("1.0", tk.END)
        self.explanation_text.insert("1.0", messages.EXPLANATION_PLACEHOLDER)
        self.explanation_text.configure(state=tk.DISABLED)

    def _build_explanation(self, row: int) -> None:
        """Build the explanation card: heading, legend and read-only text.

        Inputs: row : the grid row for the card.
        Output: none.
        """
        card = ttk.Frame(self.frame, style="Card.TFrame", padding=theme.PADDING)
        card.grid(row=row, column=0, sticky="nsew", padx=(8, 0))
        card.columnconfigure(0, weight=1)
        card.rowconfigure(2, weight=1)
        ttk.Label(card, text=messages.EXPLANATION_HEADING,
                  style="CardHeading.TLabel").grid(row=0, column=0, sticky="w")
        self._build_legend(card, row=1)

        self.explanation_text = tk.Text(card, height=8, **theme.text_widget_options())
        self.explanation_text.grid(row=2, column=0, sticky="nsew", pady=(6, 0))
        scroll = ttk.Scrollbar(card, command=self.explanation_text.yview)
        scroll.grid(row=2, column=1, sticky="ns", pady=(6, 0))
        self.explanation_text.configure(yscrollcommand=scroll.set)
        highlight.configure_tags(self.explanation_text)

    @staticmethod
    def _build_legend(parent: ttk.Frame, row: int) -> None:
        """Build the two-swatch legend explaining the highlight colours.

        Inputs: parent : the frame to place the legend in; row : its grid row.
        Output: none.
        """
        legend = ttk.Frame(parent, style="Card.TFrame")
        legend.configure(borderwidth=0)
        legend.grid(row=row, column=0, sticky="w", pady=(4, 0))
        phishing_colour, legitimate_colour = highlight.legend_colours()
        for colour, text in ((phishing_colour, messages.LEGEND_PHISHING),
                             (legitimate_colour, messages.LEGEND_LEGITIMATE)):
            tk.Label(legend, text="    ", background=colour).pack(
                side="left", padx=(0, 6))
            ttk.Label(legend, text=text, style="CardMuted.TLabel").pack(
                side="left", padx=(0, 18))
