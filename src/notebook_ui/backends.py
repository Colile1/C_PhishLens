# backends.py
# Purpose: two ways to score and explain an email from a notebook: in-process, or via the API.
# Author: Colile Sibanda

from __future__ import annotations

from typing import Protocol

import requests

from phishlens.config.loader import load_config
from phishlens.config.schema import Config


class Backend(Protocol):
    """Anything that turns email text into the `/explain` response shape."""

    def analyse(self, text: str) -> dict: ...


class LocalBackend:
    """Run the served model inside the notebook's own Python process.

    Uses the same `ModelService` the API serves from, so a verdict here is the
    verdict the API would give. Needs no server and no network: only the
    committed checkpoint, vocabulary and metrics file.

    Inputs: config : validated Config; defaults to `configs/default.yaml`.
    Output: a backend whose `analyse(text)` returns the `/explain` shape.
    """

    def __init__(self, config: Config | None = None) -> None:
        from phishlens.serving.model_service import MODEL_ID, load_model_service

        self._config = config if config is not None else load_config()
        self._service = load_model_service(self._config)
        self._model_id = MODEL_ID

    def analyse(self, text: str) -> dict:
        """Score and explain one email.

        Inputs: text : raw email subject and body.
        Output: dict with label, probability, threshold, model_id, text (the
        normalised string the offsets index into), tokens, method, caveat.

        Raises the serving layer's EmptyTextError or TextTooLongError, whose
        messages name the problem.
        """
        from phishlens.serving.routes import EXPLANATION_CAVEAT

        result, normalised, tokens = self._service.explain(text)
        return {
            "label": result.label,
            "probability": result.probability,
            "threshold": result.threshold,
            "model_id": self._model_id,
            "text": normalised,
            "tokens": [
                {"token": t.token, "weight": t.weight, "start": t.start, "end": t.end}
                for t in tokens
            ],
            "method": self._service.explain_method,
            "caveat": EXPLANATION_CAVEAT,
        }


class ApiBackend:
    """Call a running PhishLens API, local or hosted, over HTTP.

    Inputs: base_url : for example `http://127.0.0.1:8000`; timeout_seconds :
    per request. Explanations can take up to a minute on a small host.
    Output: a backend whose `analyse(text)` returns the `/explain` response.
    """

    def __init__(self, base_url: str, timeout_seconds: float = 120.0) -> None:
        self._url = base_url.rstrip("/") + "/explain"
        self._timeout = timeout_seconds

    def analyse(self, text: str) -> dict:
        """POST the text to `/explain` and return the decoded response.

        Inputs: text : raw email subject and body.
        Output: the `/explain` response as a dict.

        Raises RuntimeError with the API's own reason, or naming the URL when
        the API cannot be reached.
        """
        try:
            response = requests.post(self._url, json={"text": text}, timeout=self._timeout)
        except requests.RequestException as error:
            raise RuntimeError(
                f"Cannot reach the PhishLens API at {self._url}: {error.__class__.__name__}. "
                "Start it with `python run_api.py`, or use LocalBackend()."
            ) from None
        if response.status_code != 200:
            raise RuntimeError(f"The API refused the request ({response.status_code}): {_detail(response)}")
        return response.json()


def _detail(response: requests.Response) -> str:
    """Pull the readable reason out of an error response.

    Inputs: response : a non-200 response from the API.
    Output: the `detail` message, field-prefixed for validation errors.
    """
    try:
        detail = response.json().get("detail")
    except ValueError:
        return response.text[:200]
    if isinstance(detail, list):
        return " ".join(f"{item.get('loc', ['?'])[-1]}: {item.get('msg', '')}" for item in detail)
    return str(detail)
