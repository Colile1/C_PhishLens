# Dockerfile
# Purpose: serving image for the PhishLens API and browser UI on Render.
# Author: Colile Sibanda
#
# Built by Render from a clone of the GitHub repository, with the repository
# root as build context (render.yaml). Only the served model, its vocabulary
# and its threshold file are copied in; data and notebooks never enter the
# image.

FROM python:3.11-slim

RUN useradd --create-home --uid 1000 user
WORKDIR /app

COPY deploy/requirements-serve.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 \
 && pip install --no-cache-dir -r requirements-serve.txt

COPY --chown=user src/phishlens src/phishlens
COPY --chown=user src/web src/web
COPY --chown=user run_api.py .
COPY --chown=user configs/default.yaml configs/default.yaml
COPY --chown=user deploy/render/local.yaml configs/local.yaml
COPY --chown=user artifacts/models/text_cnn_best.pt artifacts/models/vocabulary.json artifacts/models/
COPY --chown=user artifacts/metrics/m1_text_cnn.json artifacts/metrics/

USER user
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1
EXPOSE 10000

CMD ["python", "run_api.py"]
