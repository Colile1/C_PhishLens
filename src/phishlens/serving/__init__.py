# __init__.py
# Purpose: public entry point for the serving package.
# Author: Colile Sibanda

from phishlens.serving.app import create_app
from phishlens.serving.model_service import ModelService, Prediction, load_model_service

__all__ = [
    "create_app",
    "ModelService",
    "Prediction",
    "load_model_service",
]
