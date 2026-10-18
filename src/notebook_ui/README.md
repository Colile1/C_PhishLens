# src/notebook_ui/

The notebook client: a third way to use PhishLens, beside the Tkinter app in
`src/desktop/` and the browser UI in `src/web/`. It is a presentation layer and
holds no model logic. `notebooks/07_analyst_ui.ipynb` is the page that uses it.

A notebook committed with its outputs is **readable** in any static viewer
(GitHub included), because results render as plain HTML. Opened in Jupyter with
a running kernel, it is **usable**: call `analyse(text)` on any email, or use
the interactive panel from `launch()`.

## Files

| File | Responsibility |
|------|----------------|
| `__init__.py` | `analyse(text, backend=None)` displays the result; `launch(backend=None)` shows the interactive panel. Both default to one shared `LocalBackend` |
| `backends.py` | `LocalBackend` runs the served model in the notebook's own process through the same `ModelService` the API uses (no server, no network). `ApiBackend(base_url)` calls a running API instead, local or hosted |
| `render.py` | `render_result(result)` : the verdict line, highlighted normalised text and a strongest-signals table, as HTML. Rendering only |
| `widget.py` | The ipywidgets panel: text box, Analyse and Clear buttons, rendered result |

## Behaviour it gets right

- Every piece of email text is HTML-escaped before rendering, so a pasted email
  cannot inject markup or script into the notebook.
- Direction is shown by colour, by **bold** (towards phishing) or *italic*
  (towards legitimate), and in words in the table, so it survives viewers that
  strip colour.
- Highlights index into the normalised text the model scored, never the raw input.
- Errors name the reason (blank text, text too long, API unreachable).
- The caution that the model assists triage and does not decide.

## Usage

```python
from notebook_ui import analyse, launch, ApiBackend

analyse("URGENT: verify your password at http://example.test/login ...")
launch()                                   # interactive panel, local model
analyse(text, ApiBackend("https://phishlens-vdc9.onrender.com"))   # hosted API
```

Verified by `tests/unit/test_notebook_ui.py`.
