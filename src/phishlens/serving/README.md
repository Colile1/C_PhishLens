# src/phishlens/serving/

The FastAPI service. This is the integration layer: it imports the core, loads
the model once at startup, and exposes it over HTTP. It contains no model logic
of its own, which is what allows the desktop app to be swapped for a hosted
deployment without touching anything else.

Half of the 15 points live here, the other half in `src/desktop/`.

## Planned files

| File | Responsibility |
|------|----------------|
| `app.py` | `create_app(config) -> FastAPI` : application factory. Wires routes, loads the model, vocabulary and threshold once in a startup handler. No request handling logic |
| `routes.py` | The four endpoints. `GET /health`, `GET /metadata`, `POST /predict`, `POST /explain`. Each is a thin translation between the schema and a core call |
| `schemas.py` | Pydantic request and response models: `PredictRequest(text)`, `PredictResponse(label, probability, threshold, model_id)`, `ExplainResponse(tokens, method, caveat)`. Validation lives here, so a bad request never reaches the core |
| `model_service.py` | `ModelService` : holds the loaded model, vocabulary and threshold. `predict(text) -> Prediction` and `explain(text) -> list[TokenWeight]`. The single place the core is called from |
| `errors.py` | Exception handlers turning core errors into responses that name the problem. No stack traces in responses |

## Endpoints

| Method | Path | Returns |
|--------|------|---------|
| `GET` | `/health` | Service up, model loaded yes/no |
| `GET` | `/metadata` | Model id, architecture, decision threshold, seed, training commit hash, vocabulary hash, dataset version |
| `POST` | `/predict` | Label, probability, the threshold used, the model id |
| `POST` | `/explain` | Per-token weights with character offsets, the method name, and the caveat that the explanation is local to this prediction |

## Rules

- The model is loaded **once**, at startup, not per request.
- Startup fails loudly if the vocabulary hash in the model metadata does not
  match the vocabulary file on disk. A mismatch would serve confident nonsense.
- Empty or whitespace-only text returns 422 naming the `text` field. Input over
  `config.serving.max_input_characters` returns 413 stating the limit.
- Every response carries the model id, so a screenshot taken during the demo can
  be traced to a commit.
- No endpoint retrains, refits or writes to `artifacts/`.

## Integration

Imports `config`, `features`, `models` and `explain`. **Nothing in the core
imports this folder.** Started by `run_api.py` at the repository root, which
reads the config and calls `create_app`.

Done when: `tests/unit/test_api.py` passes with the FastAPI `TestClient` : health
reports the model loaded, empty text is refused with a 422 naming `text`,
`/predict` returns a probability in [0, 1], and `/explain` returns tokens whose
offsets index correctly into the submitted text.
