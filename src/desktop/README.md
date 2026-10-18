# src/desktop/

The Tkinter desktop application. This is a presentation layer and holds no
model logic at all: it sends text to the FastAPI service and renders what comes
back. Keeping the model out of this folder means the app can run against a
local or a hosted API unchanged, and the model can be tested without a GUI.

## Files

| File | Responsibility |
|------|----------------|
| `app.py` | `PhishLensApp(api_base_url)` : builds the window and wires events. Holds no analysis logic |
| `api_client.py` | `predict`, `explain` and `health` : HTTP calls to the service, with timeouts, and errors translated into readable messages. The only file in this folder that knows the API exists |
| `highlight.py` | `render_attributions(widget, text, tokens)` : paints token backgrounds by attribution sign and magnitude in a Tkinter `Text` widget. Pure rendering |
| `messages.py` | Every user-facing string in one place: verdict wording, error text, the caution that the model assists triage and does not decide |
| `theme.py` | Palette, fonts, ttk styles, and Windows high-DPI awareness so text renders sharp on scaled displays |
| `header_bar.py` | Title bar with a live API status indicator (polls `/health`; click to re-check) |
| `input_panel.py` | Input card: text area, word count, Paste / Open file / Clear / Analyse (Ctrl+Enter) |
| `results_panel.py` | Right-hand pane: verdict card, caution line, highlighted explanation, strongest signals |
| `verdict_card.py` | Verdict banner and probability meter with the decision threshold marked |
| `signals_panel.py` / `signals.py` | The strongest tokens each way; `signals.py` is the pure ranking logic |
| `email_file.py` | Reads a `.eml` as subject and body only (headers are dropped on purpose: they carry the corpus artefacts the model is kept away from) or a `.txt` verbatim |

## Behaviour it gets right

- The verdict shows the label **and** the probability, never a bare "phishing".
- The explanation highlights tokens that pushed towards phishing and tokens that
  pushed away, distinguishably, with a legend.
- A visible caution: the model assists triage and does not decide.
- If the API is down, slow or returns an error, the user sees a sentence naming
  what failed and what to do. Never a traceback.
- Empty or whitespace-only input is refused before a request is sent.

## Running it

Start the API (`python run_api.py`), then `python run_app.py`. To use the hosted
API instead, set `desktop.api_base_url` in `configs/local.yaml`.

Must **not** import `src/phishlens/models`, `training` or `data`. Verified by
the `tests/unit/test_desktop_*.py` files. Screenshots are in `docs/report/`.
