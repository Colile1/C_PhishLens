# app.py
# Purpose: application factory. Wires routes and loads the model once at startup.
# Author: Colile Sibanda

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from phishlens.config.schema import Config
from phishlens.serving.errors import register_error_handlers
from phishlens.serving.model_service import load_model_service
from phishlens.serving.routes import router

APP_TITLE = "PhishLens API"

# The browser front end: static files, no model logic.
WEB_DIR = Path(__file__).resolve().parents[2] / "web"
WEB_MOUNT = "/ui"


def create_app(config: Config) -> FastAPI:
    """Build the FastAPI app: wire routes, load the model once at startup.

    The model, vocabulary and threshold are loaded inside the lifespan
    context, not at import time and not per request (serving/README.md).
    A failure here (missing checkpoint, missing vocabulary, missing metrics
    file) fails startup loudly rather than serving a half-initialised app;
    `app.state.model_service` simply stays unset and every route reports a
    503 via `ModelNotLoadedError` until it is fixed and the process restarted.

    The browser front end in `src/web/` is mounted as static files at
    `/ui/` when that folder exists; it adds no endpoint to the four.

    Inputs: config : validated Config.
    Output: a FastAPI instance ready to be served by `run_api.py` or wrapped
    in a `TestClient`.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model_service = load_model_service(config)
        yield
        app.state.model_service = None

    app = FastAPI(title=APP_TITLE, lifespan=lifespan)
    app.state.model_service = None
    app.include_router(router)
    register_error_handlers(app)
    if WEB_DIR.is_dir():
        app.mount(WEB_MOUNT, StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
