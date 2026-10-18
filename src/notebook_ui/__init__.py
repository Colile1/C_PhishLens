# __init__.py
# Purpose: the notebook client's public entry points: analyse(), launch() and the backends.
# Author: Colile Sibanda

from __future__ import annotations

from notebook_ui.backends import ApiBackend, Backend, LocalBackend
from notebook_ui.render import render_result, top_signals

_default_backend: Backend | None = None


def analyse(text: str, backend: Backend | None = None):
    """Score and explain one email and display the result in the notebook.

    The rendered output is plain HTML, so a notebook committed with these
    outputs stays readable in a static viewer.

    Inputs: text : raw email subject and body; backend : optional, defaults to
    a LocalBackend created once and reused.
    Output: an IPython HTML object, which Jupyter displays.
    """
    from IPython.display import HTML

    return HTML(render_result(_resolve(backend).analyse(text)))


def launch(backend: Backend | None = None):
    """Show the interactive input panel (needs a live Jupyter kernel).

    Inputs: backend : optional, defaults to the shared LocalBackend.
    Output: the ipywidgets panel.
    """
    from notebook_ui.widget import launch as _launch

    return _launch(_resolve(backend))


def _resolve(backend: Backend | None) -> Backend:
    """Return the given backend, or the shared LocalBackend, loading it once.

    Inputs: backend : a backend or None.
    Output: a backend ready to analyse.
    """
    global _default_backend
    if backend is not None:
        return backend
    if _default_backend is None:
        _default_backend = LocalBackend()
    return _default_backend


__all__ = ["analyse", "launch", "LocalBackend", "ApiBackend", "render_result", "top_signals"]
