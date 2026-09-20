# routes.py
# Purpose: the four endpoints, each a thin translation between schema and ModelService.
# Author: Colile Sibanda

from __future__ import annotations

from fastapi import APIRouter, Request

from phishlens.explain.attributions import PHISHING_IS_POSITIVE
from phishlens.serving.errors import ModelNotLoadedError
from phishlens.serving.model_service import ARCHITECTURE, MODEL_ID, ModelService
from phishlens.serving.schemas import (
    ExplainResponse,
    HealthResponse,
    MetadataResponse,
    PredictRequest,
    PredictResponse,
    TokenWeightResponse,
)

EXPLANATION_CAVEAT = (
    "This explanation is local to this one prediction (Ribeiro et al., 2016): "
    "it describes why this email was scored as it was, not the model's "
    "behaviour in general."
)

router = APIRouter()


def _get_model_service(request: Request) -> ModelService:
    """Fetch the ModelService the startup handler attached to app.state.

    Inputs: request : the current request, carrying `app.state`.
    Output: the loaded ModelService.

    Raises ModelNotLoadedError, turned into a 503, if startup has not
    finished loading the model.
    """
    service = getattr(request.app.state, "model_service", None)
    if service is None:
        raise ModelNotLoadedError(
            "The model is not loaded yet. The service is starting up or "
            "startup failed; check the server logs."
        )
    return service


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Report whether the service is up and the model is loaded.

    Inputs: request : the current request.
    Output: HealthResponse with model_loaded set from `app.state`.
    """
    loaded = getattr(request.app.state, "model_service", None) is not None
    return HealthResponse(status="ok", model_loaded=loaded)


@router.get("/metadata", response_model=MetadataResponse)
def metadata(request: Request) -> MetadataResponse:
    """Report the identity and provenance of the currently served model.

    Inputs: request : the current request.
    Output: MetadataResponse naming the model, threshold, seed, training
    commit, vocabulary hash and dataset version, so any demo screenshot
    traces back to the run that produced the served model.

    Raises ModelNotLoadedError, turned into a 503, before startup finishes.
    """
    service = _get_model_service(request)
    return MetadataResponse(
        model_id=MODEL_ID,
        architecture=ARCHITECTURE,
        threshold=service.threshold,
        seed=service.seed,
        training_commit=service.training_commit,
        vocabulary_hash=service.vocabulary_hash,
        dataset_version=service.dataset_version,
    )


@router.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    """Classify one email as phishing or legitimate.

    Inputs: payload : validated PredictRequest carrying `text`; request : the
    current request.
    Output: PredictResponse with probability in [0, 1].

    Raises EmptyTextError (422) or TextTooLongError (413) via ModelService;
    ModelNotLoadedError (503) before startup finishes.
    """
    service = _get_model_service(request)
    result = service.predict(payload.text)
    return PredictResponse(
        label=result.label,
        probability=result.probability,
        threshold=result.threshold,
        model_id=MODEL_ID,
    )


@router.post("/explain", response_model=ExplainResponse)
def explain(payload: PredictRequest, request: Request) -> ExplainResponse:
    """Classify and explain one email with per-token LIME attributions.

    Sign convention: a positive weight pushes the prediction towards
    phishing, matching `explain.attributions.PHISHING_IS_POSITIVE`.

    Inputs: payload : validated PredictRequest carrying `text`; request : the
    current request.
    Output: ExplainResponse carrying `text`, the normalised string every
    token's offset indexes into, with the caveat that the
    explanation is local to this one prediction.

    Raises EmptyTextError (422) or TextTooLongError (413) via ModelService;
    ModelNotLoadedError (503) before startup finishes.
    """
    assert PHISHING_IS_POSITIVE  # sign convention asserted once, at the boundary
    service = _get_model_service(request)
    result, normalised_text, tokens = service.explain(payload.text)
    return ExplainResponse(
        label=result.label,
        probability=result.probability,
        threshold=result.threshold,
        model_id=MODEL_ID,
        text=normalised_text,
        tokens=[
            TokenWeightResponse(token=t.token, weight=t.weight, start=t.start, end=t.end)
            for t in tokens
        ],
        method=service.explain_method,
        caveat=EXPLANATION_CAVEAT,
    )
