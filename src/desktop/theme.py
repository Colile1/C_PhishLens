# theme.py
# Purpose: the desktop app's palette, fonts, ttk styles and high-DPI setup.
# Author: Colile Sibanda

from __future__ import annotations

import sys
import tkinter as tk
from tkinter import ttk

FONT_FAMILY = "Segoe UI"
MONO_FAMILY = "Consolas"

FONT_BODY = (FONT_FAMILY, 10)
FONT_SMALL = (FONT_FAMILY, 9)
FONT_HEADING = (FONT_FAMILY, 11, "bold")
FONT_TITLE = (FONT_FAMILY, 16, "bold")
FONT_VERDICT = (FONT_FAMILY, 18, "bold")
FONT_TEXT = (FONT_FAMILY, 11)
FONT_MONO = (MONO_FAMILY, 10)

# Neutral surfaces.
BACKGROUND = "#f3f4f7"
SURFACE = "#ffffff"
BORDER = "#d7dbe3"
TEXT = "#1f2430"
MUTED = "#5f6776"
HEADER = "#1b2a4a"
ACCENT = "#2f5bd3"
ACCENT_ACTIVE = "#2447a8"

# Verdict colours. Red towards phishing, blue towards legitimate, matching the
# token highlights (highlight.py), so one colour means one direction app-wide.
PHISHING = "#c62828"
LEGITIMATE = "#1f5fbf"
NEUTRAL = "#7a8394"

# API status indicator: light tints, since it sits on the dark header.
STATUS_ONLINE = "#7ee08a"
STATUS_LOADING = "#ffc566"
STATUS_OFFLINE = "#ff8a80"

PADDING = 12


def enable_high_dpi() -> None:
    """Make Windows render the window at native resolution, not bitmap-scaled.

    Without this, Tk on a scaled display (125%, 150%) is drawn at 96 DPI and
    stretched, which is what makes it look blurry. Must run before the Tk
    root is created. A no-op on other platforms or older Windows builds.

    Inputs: none.
    Output: none.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        return


def apply_styles(root: tk.Misc) -> None:
    """Register the named ttk styles the window's widgets use.

    Inputs: root : the Tk root the styles belong to.
    Output: none; configures the root's ttk.Style and background.
    """
    root.configure(background=BACKGROUND)
    style = ttk.Style(root)
    style.theme_use("clam")

    style.configure(".", font=FONT_BODY, background=BACKGROUND, foreground=TEXT)
    style.configure("TFrame", background=BACKGROUND)
    style.configure("Card.TFrame", background=SURFACE, relief="solid",
                    borderwidth=1, bordercolor=BORDER)
    style.configure("Header.TFrame", background=HEADER)

    style.configure("TLabel", background=BACKGROUND, foreground=TEXT)
    style.configure("Card.TLabel", background=SURFACE)
    style.configure("CardHeading.TLabel", background=SURFACE, font=FONT_HEADING)
    style.configure("CardMuted.TLabel", background=SURFACE, foreground=MUTED,
                    font=FONT_SMALL)
    style.configure("Title.TLabel", background=HEADER, foreground="#ffffff",
                    font=FONT_TITLE)
    style.configure("Subtitle.TLabel", background=HEADER, foreground="#c5cde0",
                    font=FONT_SMALL)
    style.configure("Status.TLabel", background=BACKGROUND, foreground=MUTED,
                    font=FONT_SMALL, padding=(PADDING, 4))
    style.configure("StatusError.TLabel", background="#fdecea", foreground=PHISHING,
                    font=(FONT_FAMILY, 10, "bold"), padding=(PADDING, 6))

    style.configure("TButton", padding=(12, 6))
    style.configure("Accent.TButton", background=ACCENT, foreground="#ffffff",
                    font=(FONT_FAMILY, 10, "bold"), padding=(18, 7),
                    bordercolor=ACCENT)
    style.map("Accent.TButton",
              background=[("disabled", "#9aaad6"), ("active", ACCENT_ACTIVE)],
              foreground=[("disabled", "#eef1f8")])


def text_widget_options() -> dict:
    """Return the shared look for the input and explanation Text widgets.

    Inputs: none.
    Output: keyword options for tk.Text.
    """
    return {
        "wrap": "word",
        "font": FONT_TEXT,
        "background": SURFACE,
        "foreground": TEXT,
        "relief": "flat",
        "borderwidth": 0,
        "highlightthickness": 1,
        "highlightbackground": BORDER,
        "highlightcolor": ACCENT,
        "padx": 10,
        "pady": 8,
        "spacing1": 2,
        "spacing3": 2,
        "insertbackground": TEXT,
    }
