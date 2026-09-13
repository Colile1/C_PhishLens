# __init__.py
# Purpose: public entry point for the models package.
# Author: Colile Sibanda

from phishlens.models.baseline import build_baseline
from phishlens.models.text_cnn import TextCNN, build_text_cnn, count_parameters

__all__ = [
    "build_baseline",
    "TextCNN",
    "build_text_cnn",
    "count_parameters",
]
