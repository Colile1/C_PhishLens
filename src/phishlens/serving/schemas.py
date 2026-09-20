# schemas.py
# Purpose: request and response contracts for the four endpoints.
# Author: Colile Sibanda

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from phishlens.serving.errors import EmptyTextError


class PredictRequest(BaseModel):
    """Body of POST /predict and POST /explain.

    Inputs: text : the raw email subject and/or body to classify.
    Output: validated PredictRequest; blank or whitespace-only text is
    rejected before it reaches the model.
    """

    text: str = Field(..., description="Raw email subject and/or body text.")

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        """Reject blank or whitespace-only text with a field-naming error.

        Inputs: value : the raw `text` field as submitted.
        Output: the value unchanged, if non-blank.

        Raises EmptyTextError, which routes.py turns into a 422 naming `text`.
        """
        if not value.strip():
            raise EmptyTextError("text must not be blank or whitespace-only.")
        return value


class PredictResponse(BaseModel):
    """Body returned by POST /predict.

    Inputs: label : "phishing" or "legitimate"; probability : P(phishing) in
    [0, 1]; threshold : the decision threshold the label was derived from;
    model_id : the model name, so a screenshot traces to a commit.
    Output: immutable PredictResponse.
    """

    label: str
    probability: float
    threshold: float
    model_id: str


class TokenWeightResponse(BaseModel):
    """One token occurrence with its signed attribution weight.

    Inputs: token, weight, start, end : see explain.attributions.TokenWeight.
    Output: immutable TokenWeightResponse.
    """

    token: str
    weight: float
    start: int
    end: int


class ExplainResponse(BaseModel):
    """Body returned by POST /explain.

    Inputs: label, probability, threshold, model_id : as in PredictResponse,
    so one call to /explain carries the full prediction too; text : the
    normalised text LIME actually explained, which every token's start/end
    offset indexes into : a client cannot highlight
    correctly from offsets alone without it; tokens : every token occurrence
    in `text`, in order, with a weight of 0.0 where LIME assigned none;
    method : the XAI method name; caveat : the fixed reminder that this
    explanation is local to this one prediction.
    Output: immutable ExplainResponse.
    """

    label: str
    probability: float
    threshold: float
    model_id: str
    text: str
    tokens: list[TokenWeightResponse]
    method: str
    caveat: str


class HealthResponse(BaseModel):
    """Body returned by GET /health.

    Inputs: status : "ok"; model_loaded : whether startup finished loading
    the model successfully.
    Output: immutable HealthResponse.
    """

    status: str
    model_loaded: bool


class MetadataResponse(BaseModel):
    """Body returned by GET /metadata.

    Inputs: model_id, architecture, threshold, seed, training_commit,
    vocabulary_hash, dataset_version : provenance fields so a demo screenshot
    can be traced back to the run that produced the served model.
    Output: immutable MetadataResponse.
    """

    model_id: str
    architecture: str
    threshold: float
    seed: int
    training_commit: str
    vocabulary_hash: str
    dataset_version: str
