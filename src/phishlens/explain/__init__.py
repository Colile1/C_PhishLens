# __init__.py
# Purpose: public entry point for the explain package.
# Author: Colile Sibanda

from phishlens.explain.attributions import (
    PHISHING_IS_POSITIVE,
    TokenWeight,
    align_to_tokens,
    top_tokens,
)
from phishlens.explain.lime_explainer import CLASS_NAMES, Explanation, explain_text
from phishlens.explain.predict_fn import (
    PHISHING_COLUMN,
    LEGITIMATE_COLUMN,
    ProbabilityFn,
    make_baseline_predict_fn,
    make_text_cnn_predict_fn,
)
from phishlens.explain.stability import StabilityReport, stability_report

__all__ = [
    "TokenWeight",
    "align_to_tokens",
    "top_tokens",
    "PHISHING_IS_POSITIVE",
    "Explanation",
    "explain_text",
    "CLASS_NAMES",
    "ProbabilityFn",
    "make_text_cnn_predict_fn",
    "make_baseline_predict_fn",
    "PHISHING_COLUMN",
    "LEGITIMATE_COLUMN",
    "StabilityReport",
    "stability_report",
]
