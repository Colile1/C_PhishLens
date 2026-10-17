# Dockerfile
# Purpose: serving image for the PhishLens API and browser UI on Hugging Face Spaces.
# Author: Colile Sibanda
#
# Built from the folder scripts/deploy_space.py assembles, not from the repo
# root. CPU-only torch keeps the image small; the Space has no GPU.

FROM python:3.11-slim

# Spaces run the container as user 1000.
RUN useradd --create-home --uid 1000 user
WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 \
 && pip install --no-cache-dir -r requirements-serve.txt

COPY --chown=user . .

USER user
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1
EXPOSE 7860

CMD ["python", "run_api.py"]
