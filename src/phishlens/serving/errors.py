# errors.py
# Purpose: turn core and validation errors into responses that name the problem.
# Author: Colile Sibanda

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class EmptyTextError(ValueError):
    """Raised when the request's `text` field is blank or whitespace-only.

    Inputs: none beyond the standard ValueError message.
    Output: caught by `register_error_handlers` and turned into a 422 naming
    `text`, rather than reaching the model as an empty sequence.
    """


class TextTooLongError(ValueError):
    """Raised when `text` exceeds `config.serving.max_input_characters`.

    Inputs: length : the submitted character count; limit : the configured
    maximum.
    Output: caught by `register_error_handlers` and turned into a 413 stating
    both numbers.
    """

    def __init__(self, length: int, limit: int) -> None:
        self.length = length
        self.limit = limit
        super().__init__(
            f"text is {length} characters, which exceeds the {limit}-character limit."
        )


class ModelNotLoadedError(RuntimeError):
    """Raised when a route is called before startup finished loading the model.

    Inputs: none beyond the standard RuntimeError message.
    Output: caught by `register_error_handlers` and turned into a 503, since
    the service is up but not yet able to serve a prediction.
    """


def register_error_handlers(app: FastAPI) -> None:
    """Attach exception handlers so no response ever carries a raw traceback.

    Inputs: app : the FastAPI instance to register handlers on.
    Output: none; mutates `app` in place.
    """

    @app.exception_handler(EmptyTextError)
    async def _handle_empty_text(request: Request, exc: EmptyTextError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"detail": [{"loc": ["body", "text"], "msg": str(exc)}]},
        )

    @app.exception_handler(TextTooLongError)
    async def _handle_text_too_long(request: Request, exc: TextTooLongError) -> JSONResponse:
        return JSONResponse(
            status_code=413,
            content={"detail": [{"loc": ["body", "text"], "msg": str(exc)}]},
        )

    @app.exception_handler(ModelNotLoadedError)
    async def _handle_model_not_loaded(
        request: Request, exc: ModelNotLoadedError
    ) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})
