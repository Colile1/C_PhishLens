---
title: PhishLens
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Phishing email classifier with token-level explanations
---

# PhishLens

A Text-CNN that classifies an email as phishing or legitimate from its subject
and body text alone, and highlights the words that drove the decision (LIME).
Built for ITRI625 (North-West University, 2026) by Colile Sibanda.

- **Browser UI:** the Space opens on it (`/ui/`).
- **API:** `GET /health`, `GET /metadata`, `POST /predict`, `POST /explain`
  with body `{"text": "..."}`. Interactive docs at `/docs`.
- **Source, notebooks and report:** https://github.com/Colile1/C_PhishLens

The model assists triage: it does not decide. It was trained on pooled public
corpora and scores lower on mail from sources it has not seen; the report
measures this. Nothing you paste is stored, but do not paste real personal
email into a public demo.
