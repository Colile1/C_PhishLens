# src/phishlens/serving/

The FastAPI service. This is the integration layer: it imports the core, loads
the model once at startup, and exposes it over HTTP. It contains no model logic
of its own, which is what lets the desktop app, the browser UI and a hosted
deployment all use it unchanged.

## Files

| File | Responsibility |
|------|----------------|
| `app.py` | `create_app(config) -> FastAPI` : wires the routes, loads the model once in the lifespan handler, adds the security middleware, mounts the browser UI at `/ui/` and redirects `/` there |
| `routes.py` | The four endpoints, each a thin translation between a schema and the model service |
| `schemas.py` | Pydantic request and response models. Validation lives here, so a bad request never reaches the core |
| `model_service.py` | `ModelService` : holds the loaded model, vocabulary and threshold; `predict(text)` and `explain(text)`. The single place the core is called from |
| `loading.py` | Load the vocabulary, the checkpoint and the validation-selected threshold; hash the vocabulary (line endings normalised, so the hash is the same on every platform) |
| `errors.py` | Exception handlers that turn core errors into responses naming the problem. No stack traces in responses |
| `security.py` | Security headers on every response, a strict content security policy on the UI, and per-client rate limits on the scoring endpoints |

## Endpoints

| Method | Path | Returns |
|--------|------|---------|
| `GET` | `/health` | Service up, model loaded yes/no |
| `GET` | `/metadata` | Model id, architecture, decision threshold, seed, training commit, vocabulary hash, dataset version |
| `POST` | `/predict` | Label, probability, the threshold used, the model id |
| `POST` | `/explain` | The prediction plus per-token weights with character offsets into the normalised text, the method name, and the caveat that the explanation is local |

## Rules

- The model is loaded **once**, at startup, not per request.
- Empty or whitespace-only text returns 422 naming the `text` field. Input over
  `config.serving.max_input_characters` returns 413 stating the limit.
- Too many scoring requests from one client return 429 with a `Retry-After`
  header; limits are in `config.serving`.
- Every response carries the model id, so a screenshot can be traced to a commit.
- No endpoint retrains, refits or writes to `artifacts/`.

## Integration

Imports `config`, `features`, `models` and `explain`. **Nothing in the core
imports this folder.** Started by `run_api.py` at the repository root.

Verified by `tests/unit/test_api.py` and `tests/unit/test_web_ui.py`.
