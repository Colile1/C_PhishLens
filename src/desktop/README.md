# src/desktop/

The Tkinter desktop application : 15 of the 70 points, together with
the API it talks to. This is the presentation layer and holds no model logic at
all: it sends text to the FastAPI service and renders what comes back.

Keeping the model out of this folder is what lets the app be judged on whether
it works, and lets the model be tested without a GUI.

## Files

| File | Responsibility |
|------|----------------|
| `app.py` | `PhishLensApp(api_base_url: str)` : builds the window: input text area, Analyse button, verdict panel, explanation panel, status bar. Holds no analysis logic |
| `api_client.py` | `predict(text: str) -> Prediction` and `explain(text: str) -> list[TokenWeight]` : HTTP calls to the service, with timeouts, and errors translated into readable messages. The only file in this folder that knows the API exists |
| `highlight.py` | `render_attributions(widget, text: str, weights: list[TokenWeight]) -> None` : paints token backgrounds by attribution sign and magnitude in a Tkinter `Text` widget. Pure rendering, no HTTP |
| `messages.py` | User-facing strings in one place: verdict wording, error text, the caution line that the model is an aid and not a decision |
| `theme.py` | Palette, fonts, ttk styles, and Windows high-DPI awareness so text renders sharp on scaled displays |
| `header_bar.py` | Title bar with a live API status indicator (polls `/health`; click to re-check) |
| `input_panel.py` | Input card: text area, word count, Paste / Open file / Clear / Analyse (Ctrl+Enter) |
| `results_panel.py` | Right-hand pane: verdict card, caution line, highlighted explanation, strongest signals |
| `verdict_card.py` | Coloured verdict banner and probability meter with the decision threshold marked |
| `signals_panel.py` / `signals.py` | The strongest tokens each way; `signals.py` is the pure ranking logic |
| `email_file.py` | Reads a `.eml` as subject and body only (headers dropped: they are the artefact trap) or a `.txt` verbatim |

## Behaviour it must get right

- The verdict shows the label **and** the probability, never a bare "phishing".
- The explanation panel highlights tokens that pushed towards phishing and
  tokens that pushed away, distinguishably, with a legend.
- A visible caution: the model assists triage and does not decide. This is a
  claim the report also makes, so the app has to match it.
- If the API is down, slow or returns an error, the user sees a readable
  sentence naming what failed and what to do. Never a traceback in the GUI.
- Empty or whitespace-only input is refused before a request is sent.

## Integration

Depends on `requests` and on the running API. Must **not** import
`src/phishlens/models`, `training`, or `data` : if this folder can import the
model, the layering has failed and the app can no longer be demonstrated
against a hosted service.

Launched by `run_app.py` at the repository root, which does nothing but read the
config and construct `PhishLensApp`.

Done when: with the API running, pasting a phishing email shows a label, a
probability and highlighted tokens; stopping the API and clicking Analyse shows
a readable message. Screenshot of both saved to `docs/report/`.
