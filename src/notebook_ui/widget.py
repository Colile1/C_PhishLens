# widget.py
# Purpose: an interactive input panel for Jupyter: text box, Analyse button, rendered result.
# Author: Colile Sibanda

from __future__ import annotations

from notebook_ui.backends import Backend
from notebook_ui.render import render_result


def launch(backend: Backend):
    """Show a text box and an Analyse button that render results in place.

    Needs a live Jupyter kernel: a static viewer such as GitHub shows only the
    text fallback. `notebook_ui.analyse(text)` gives the same result without
    widgets.

    Inputs: backend : a LocalBackend or ApiBackend.
    Output: the ipywidgets container, which Jupyter displays.

    Raises ImportError naming the package to install if ipywidgets is missing.
    """
    try:
        import ipywidgets as widgets
    except ImportError:
        raise ImportError(
            "The interactive panel needs ipywidgets: pip install -r requirements.txt. "
            "notebook_ui.analyse(text) works without it."
        ) from None

    box = widgets.Textarea(
        placeholder="Paste the subject and body of an email",
        layout=widgets.Layout(width="100%", height="180px"),
    )
    button = widgets.Button(description="Analyse email", button_style="primary")
    clear = widgets.Button(description="Clear")
    output = widgets.HTML()

    def on_analyse(_):
        """Score the box's text and render it, or show the reason it failed."""
        if not box.value.strip():
            output.value = "<p><b>Enter some email text first.</b></p>"
            return
        button.disabled = True
        output.value = "<p>Analysing: LIME scores 1 000 variations of the email.</p>"
        try:
            output.value = render_result(backend.analyse(box.value))
        except (RuntimeError, ValueError) as error:
            from html import escape

            output.value = f"<p><b>Could not analyse this email:</b> {escape(str(error))}</p>"
        finally:
            button.disabled = False

    def on_clear(_):
        """Empty the text box and the result."""
        box.value = ""
        output.value = ""

    button.on_click(on_analyse)
    clear.on_click(on_clear)
    return widgets.VBox([box, widgets.HBox([button, clear]), output])
