# input_panel.py
# Purpose: the email input card: text area, word count and action buttons.
# Author: Colile Sibanda

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import ttk

from desktop import messages, theme


class InputPanel:
    """The left-hand card where the analyst pastes or loads an email.

    Holds no behaviour of its own: every button calls back into the app.

    Inputs: parent : the widget to build into; on_analyse, on_clear,
    on_paste, on_open : callbacks for the four buttons.
    Output: an InputPanel exposing `frame`, `input_text` and `analyse_button`.
    """

    def __init__(
        self,
        parent: tk.Misc,
        on_analyse: Callable[[], None],
        on_clear: Callable[[], None],
        on_paste: Callable[[], None],
        on_open: Callable[[], None],
    ) -> None:
        self.frame = ttk.Frame(parent, style="Card.TFrame", padding=theme.PADDING)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(2, weight=1)

        ttk.Label(self.frame, text=messages.INPUT_HEADING,
                  style="CardHeading.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(self.frame, text=messages.INPUT_HINT,
                  style="CardMuted.TLabel").grid(row=1, column=0, sticky="w",
                                                 pady=(0, 6))

        self.input_text = tk.Text(self.frame, undo=True, **theme.text_widget_options())
        self.input_text.grid(row=2, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(self.frame, command=self.input_text.yview)
        scroll.grid(row=2, column=1, sticky="ns")
        self.input_text.configure(yscrollcommand=scroll.set)
        self.input_text.bind("<<Modified>>", self._on_modified)

        self.count_label = ttk.Label(self.frame, style="CardMuted.TLabel",
                                     text=messages.format_word_count(""))
        self.count_label.grid(row=3, column=0, sticky="w", pady=(4, 0))

        buttons = ttk.Frame(self.frame, style="Card.TFrame")
        buttons.configure(borderwidth=0)
        buttons.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        for text, command in ((messages.PASTE_BUTTON, on_paste),
                              (messages.OPEN_BUTTON, on_open),
                              (messages.CLEAR_BUTTON, on_clear)):
            ttk.Button(buttons, text=text, command=command).pack(
                side="left", padx=(0, 6))
        self.analyse_button = ttk.Button(
            buttons, text=messages.ANALYSE_BUTTON, style="Accent.TButton",
            command=on_analyse,
        )
        self.analyse_button.pack(side="right")

    def set_text(self, text: str) -> None:
        """Replace the input area's contents.

        Inputs: text : the new contents.
        Output: none.
        """
        self.input_text.delete("1.0", tk.END)
        self.input_text.insert("1.0", text)
        self.refresh_count()

    def refresh_count(self) -> None:
        """Update the word and character count under the text area.

        Inputs: none.
        Output: none.
        """
        text = self.input_text.get("1.0", tk.END)
        self.count_label.configure(text=messages.format_word_count(text))

    def _on_modified(self, _event: tk.Event) -> None:
        """Refresh the count whenever the text changes, then re-arm the event.

        Inputs: _event : the Tk <<Modified>> event.
        Output: none.
        """
        self.refresh_count()
        self.input_text.edit_modified(False)
