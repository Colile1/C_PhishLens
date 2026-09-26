# header_bar.py
# Purpose: the title bar with the live API status indicator. Rendering only.
# Author: Colile Sibanda

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import ttk

from desktop import messages, theme


class HeaderBar:
    """Dark title bar: app name, subtitle and a clickable API status dot.

    Inputs: parent : the widget to build into; on_status_click : called when
    the analyst clicks the status indicator to re-check the API.
    Output: a HeaderBar exposing `frame` and `api_status`.
    """

    def __init__(self, parent: tk.Misc, on_status_click: Callable[[], None]) -> None:
        self.frame = ttk.Frame(parent, style="Header.TFrame",
                               padding=(theme.PADDING + 4, 10))
        ttk.Label(self.frame, text=messages.APP_TITLE, style="Title.TLabel").pack(
            side="left")
        ttk.Label(self.frame, text=messages.APP_SUBTITLE,
                  style="Subtitle.TLabel").pack(side="left", padx=(12, 0), pady=(6, 0))
        self.api_status = tk.Label(
            self.frame, text=f"●  {messages.API_CHECKING}", font=theme.FONT_SMALL,
            background=theme.HEADER, foreground="#c5cde0", cursor="hand2",
        )
        self.api_status.pack(side="right")
        self.api_status.bind("<Button-1>", lambda _event: on_status_click())

    def show_health(self, state: str, model_loaded: bool) -> None:
        """Paint the API status indicator.

        Inputs: state : "online" or "offline"; model_loaded : from /health.
        Output: none.
        """
        if state == "offline":
            text, colour = messages.API_OFFLINE, theme.STATUS_OFFLINE
        elif model_loaded:
            text, colour = messages.API_ONLINE, theme.STATUS_ONLINE
        else:
            text, colour = messages.API_LOADING, theme.STATUS_LOADING
        self.api_status.configure(text=f"●  {text}", foreground=colour)
