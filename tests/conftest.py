# conftest.py
# Purpose: shared pytest fixtures, notably the single Tk root the GUI tests use.
# Author: Colile Sibanda

from __future__ import annotations

import tkinter as tk

import pytest


@pytest.fixture(scope="session")
def tk_root():
    """Provide one Tk root for the whole test session.

    Tk cannot reliably be re-initialised after a root is destroyed in the same
    process: on this Python build the second `tk.Tk()` fails with "Can't find
    a usable tk.tcl" or `invalid command name "tcl_findLibrary"`. A per-test
    or per-module root therefore skipped most of the GUI tests silently. One
    session-scoped root, never destroyed until the session ends, keeps them
    all running.

    Widgets accept insert, tag and configure calls without `mainloop()`, so
    this is enough to test rendering and the analyse flow without showing a
    window.

    Inputs: none.
    Output: a withdrawn tk.Tk. Skips the requesting test when no display is
    available, e.g. on a headless machine.
    """
    try:
        root = tk.Tk()
    except tk.TclError as error:
        pytest.skip(f"Tkinter display unavailable: {error}")
    root.withdraw()
    yield root
    root.destroy()
