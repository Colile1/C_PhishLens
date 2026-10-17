# src/web/

The browser front end. A second client for the same API,
beside the Tkinter app in `src/desktop/`: it holds no model logic, sends text
to the FastAPI service and renders what comes back.

It is plain static HTML, CSS and JavaScript with no build step, served by the
API itself at `http://127.0.0.1:8000/ui/` (see `serving/app.py`, which mounts
this folder when it exists). Same origin as the API, so no CORS setup.

## Files

| File | Responsibility |
|------|----------------|
| `index.html` | Analyse page: email input, verdict, probability meter, highlighted explanation, strongest signals |
| `model.html` | Model page: provenance of the served model from `/metadata`, and the stated limitations |
| `guide.html` | User guide: how to use the page and what its terms mean. Static text |
| `static/css/theme.css` | The stylesheet (plain Material), plus a marked PhishLens section at the end: email input, meter, token shading, narrow-screen layout |
| `static/js/common.js` | Shared helpers from the house style: `apiCall`, `showBanner`, `buildTable`. `apiCall` also reads FastAPI's list-form error detail |
| `static/js/charts.js` | House-style helpers; used here for `fmt` and `explainer` |
| `static/js/health.js` | Polls `/health` every 8 s and fills the status badge |
| `static/js/tokens.js` | Token shading and signal ranking. Rendering only, no HTTP. Mirrors `src/desktop/highlight.py` and `signals.py` |
| `static/js/analyse.js` | Analyse page wiring: calls `/explain`, renders the result |
| `static/js/model.js` | Model page wiring: calls `/metadata` once the model is loaded |

## Behaviour it must get right

- Highlights are drawn on the normalised `text` the API returns, never on the
  raw input: the token offsets index into that string.
- Direction is never shown by colour alone: red/blue shading plus a solid or
  dotted underline, a hover title, and the word "towards ..." in the table.
- Error banners show the API's own reason (422 naming `text`, 413 stating
  the limit, 503 while loading), or say how to start the API if unreachable.
- The caution line that the model assists triage and does not decide.
- Source is ASCII-only, built with `createElement` and `textContent`.

Done when: `tests/unit/test_web_ui.py` passes (pages served, every referenced
asset resolves, the API still exposes exactly its four endpoints, source is
ASCII), and the pages show no console errors in a browser.
