# api_client.py
# Purpose: the only file in the desktop app that knows the PhishLens API exists.
# Author: Colile Sibanda

from __future__ import annotations

from dataclasses import dataclass

import requests

from desktop import messages

# These mirror the API's response schemas deliberately rather than importing
# them: src/desktop/README.md forbids this folder from importing the core, so
# that the app can be demonstrated against a hosted service.


@dataclass(frozen=True)
class TokenWeight:
    """One token occurrence with its signed attribution weight.

    Inputs: token : the token text; weight : signed attribution, positive
    towards phishing; start, end : character offsets into the explained text.
    Output: immutable TokenWeight.
    """

    token: str
    weight: float
    start: int
    end: int


@dataclass(frozen=True)
class ExplainResult:
    """One analysed email: the verdict and the explanation behind it.

    Inputs: label, probability, threshold, model_id : the prediction, as
    /predict would also return it; text : the normalised string LIME actually
    explained, which every token offset indexes into;
    tokens : token occurrences in `text`, in order; method : the XAI method
    name; caveat : the API's reminder that the explanation is local to this
    one prediction.
    Output: immutable ExplainResult.
    """

    label: str
    probability: float
    threshold: float
    model_id: str
    text: str
    tokens: tuple[TokenWeight, ...]
    method: str
    caveat: str


class ApiError(RuntimeError):
    """Raised when a request fails, carrying a message fit to show a user.

    Inputs: message : a readable sentence naming what failed and what to do.
    Output: an ApiError the app displays verbatim. No `requests` exception and
    no traceback ever reaches the GUI.
    """


class ApiClient:
    """HTTP client for the PhishLens API.

    Inputs: base_url : where the API is served; timeout : seconds to wait for
    a response before reporting the service unreachable.
    Output: an ApiClient exposing `explain`.
    """

    def __init__(self, base_url: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def explain(self, text: str) -> ExplainResult:
        """Classify one email and return the verdict with its explanation.

        One call rather than /predict followed by /explain: /explain returns
        the prediction too, so the verdict shown can never disagree with the
        explanation shown.

        Inputs: text : the raw email subject and/or body.
        Output: an ExplainResult. Raises ApiError, with a readable message,
        on any connection, timeout, HTTP or response-shape failure.
        """
        payload = self._post("/explain", text)
        return self._parse_explain(payload)

    def _post(self, path: str, text: str) -> dict:
        """POST text to an endpoint and return the decoded JSON body.

        Inputs: path : the endpoint path, e.g. "/explain"; text : the email
        text to send.
        Output: the decoded JSON body. Raises ApiError naming the failure.
        """
        url = f"{self.base_url}{path}"
        try:
            response = requests.post(
                url, json={"text": text}, timeout=self.timeout
            )
        except requests.Timeout as error:
            raise ApiError(
                messages.API_TIMEOUT.format(url=self.base_url, timeout=self.timeout)
            ) from error
        except requests.RequestException as error:
            raise ApiError(
                messages.API_UNREACHABLE.format(url=self.base_url)
            ) from error

        if response.status_code >= 400:
            raise ApiError(
                messages.API_ERROR.format(
                    status=response.status_code, detail=_detail_of(response)
                )
            )

        try:
            return response.json()
        except ValueError as error:
            raise ApiError(
                messages.API_BAD_RESPONSE.format(detail="the body was not JSON.")
            ) from error

    @staticmethod
    def _parse_explain(payload: dict) -> ExplainResult:
        """Build an ExplainResult from a decoded /explain body.

        Inputs: payload : the decoded JSON body.
        Output: an ExplainResult. Raises ApiError naming the missing or
        malformed field rather than letting a KeyError reach the GUI.
        """
        try:
            tokens = tuple(
                TokenWeight(
                    token=item["token"],
                    weight=float(item["weight"]),
                    start=int(item["start"]),
                    end=int(item["end"]),
                )
                for item in payload["tokens"]
            )
            return ExplainResult(
                label=payload["label"],
                probability=float(payload["probability"]),
                threshold=float(payload["threshold"]),
                model_id=payload["model_id"],
                text=payload["text"],
                tokens=tokens,
                method=payload["method"],
                caveat=payload["caveat"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ApiError(
                messages.API_BAD_RESPONSE.format(detail=str(error))
            ) from error


def _detail_of(response: requests.Response) -> str:
    """Extract a readable message from an error response body.

    The API's handlers return `detail` either as a string (503) or as a list
    of `{loc, msg}` entries (422, 413). Both are flattened to one sentence.

    Inputs: response : the non-2xx response.
    Output: a readable detail string, falling back to the raw body.
    """
    try:
        detail = response.json().get("detail")
    except ValueError:
        return response.text.strip() or "no detail given."

    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        parts = [
            str(entry.get("msg", entry)) if isinstance(entry, dict) else str(entry)
            for entry in detail
        ]
        return " ".join(parts) or "no detail given."
    return "no detail given."
