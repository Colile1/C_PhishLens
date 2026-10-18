# src/web/

The browser front end: a second client for the same API, beside the Tkinter
app in `src/desktop/`. It holds no model logic; it sends text to the FastAPI
service and renders what comes back.

It is plain static HTML, CSS and JavaScript with no build step and no
third-party script, served by the API itself at `/ui/` (`serving/app.py` mounts
this folder when it exists). Same origin as the API, so no CORS is needed, and
the content security policy can forbid every external source.

## Files

| File | Responsibility |
|------|----------------|
| `index.html` | Analyse page: email input, verdict, probability meter, highlighted explanation, strongest signals |
| `model.html` | Model page: provenance of the served model from `/metadata`, and the known limitations |
| `guide.html` | User guide: how to use the page and what its terms mean |
| `static/css/theme.css` | The stylesheet (a plain Material-style theme), with a PhishLens section at the end: email input, meter, token shading, narrow-screen layout |
| `static/js/common.js` | Shared helpers: `apiCall` (reads the API's own error detail, including FastAPI's list form), `showBanner`, `buildTable` |
| `static/js/charts.js` | Formatting and disclosure helpers; used here for `fmt` and `explainer` |
| `static/js/health.js` | Polls `/health` every 8 s and fills the status badge |
| `static/js/tokens.js` | Token shading and signal ranking. Rendering only. Mirrors `src/desktop/highlight.py` and `signals.py` |
| `static/js/analyse.js` | Analyse page: calls `/predict` and shows the verdict at once, then `/explain` with a seconds counter, since LIME is slow on small hosts |
| `static/js/model.js` | Model page: calls `/metadata` once the model is loaded |

## Behaviour it gets right

- Highlights are drawn on the normalised `text` the API returns, never on the
  raw input: the token offsets index into that string.
- Direction is never shown by colour alone: red/blue shading plus a solid or
  dotted underline, a hover title, and the words "towards ..." in the table.
- Error banners show the API's own reason (422 naming `text`, 413 stating the
  limit, 429 when rate-limited, 503 while loading), or say how to start the API.
- The caution that the model assists triage and does not decide.
- Source is ASCII-only, and DOM is built with `createElement` and `textContent`,
  never `innerHTML` with fetched data.

Verified by `tests/unit/test_web_ui.py`: pages are served, every referenced
asset resolves, the API still exposes exactly four endpoints, and the source is
ASCII.
