# 5. System implementation and deployment

Sections 1 to 4 concern the model and what its results mean. This section
describes the system built around it: how a request travels from an analyst to
the model and back, the three clients, how the service is hosted and protected,
and how the whole is tested. Everything described here can be inspected in the
repository and, for the hosted parts, at https://phishlens-vdc9.onrender.com.

## 5.1 Architecture

The code is split into a core library and the layers that use it. The core
(`src/phishlens/`) holds configuration, the data pipeline, features, the two
models, the training loop, evaluation and explanations, and has no knowledge
of HTTP or user interfaces. A FastAPI service (`src/phishlens/serving/`) is the
only layer that loads the trained model for live use. Three clients sit outside
the core and reach the model through that service: a browser UI, a Tkinter
desktop application and a notebook UI.

The service loads the vocabulary, the best-epoch checkpoint and the
validation-selected threshold once, at start-up. It exposes four endpoints:
`GET /health`, `GET /metadata`, `POST /predict` and `POST /explain`. A request
to `/explain` passes through the following steps:

- the security layer applies per-client and global rate limits (section 5.4);
- the request schema rejects blank text (HTTP 422, naming the `text` field) and
  text over 50 000 characters (HTTP 413, stating the limit);
- the text is normalised (lower case, HTML removed, links, addresses and
  numbers replaced by `<url>`, `<email>` and `<num>`), tokenised, mapped to
  vocabulary indices and padded or truncated to 400 tokens;
- the Text-CNN produces a logit, a sigmoid turns it into P(phishing), and the
  label follows from the threshold;
- LIME scores 1 000 perturbations of the normalised text through the same
  prediction function, and each word weight is attached to every occurrence of
  the word with character offsets into the normalised text.

The response carries the label, the probability, the threshold, the model
identifier, the normalised text, the token weights with their offsets, the
method name and the caveat that the explanation is local (Ribeiro, Singh &
Guestrin, 2016). One design rule matters more than the others: LIME, the API
and the notebooks all call a single shared prediction function. An explanation
therefore always describes the deployed pipeline rather than a
reimplementation of it.

`/metadata` reports the model identifier, the threshold, the seed, the commit
that produced the evaluation, a hash of the vocabulary and the configuration
hash of the data split, so any verdict or screenshot can be traced to the code
and data behind it.

## 5.2 Three clients

**Browser UI.** Plain HTML, CSS and JavaScript with no build step and no
third-party code, served by the API itself at `/ui/` (Figure 7). Because the
explanation can take up to a minute on the hosted instance, the page calls the
fast `/predict` first and shows the verdict at once, then shows a running
seconds counter while `/explain` completes. A Model page shows the live
provenance from `/metadata`, a user guide defines the terms, and an About page
explains how the system works, from a plain overview to expandable technical
sections (Figure 10).

**Desktop application.** Tkinter, started by `run_app.py` (Figure 8). It calls
the API on a background thread so the window never freezes during an
explanation, shows a live API status indicator, opens `.eml` files as subject
and body only (headers are dropped deliberately), and turns every failure into
a sentence that names the problem.

**Notebook UI.** `notebooks/07_analyst_ui.ipynb` with the client code in
`src/notebook_ui/` (Figure 9). `analyse(text)` renders the result as HTML, so
the committed notebook is readable in any notebook viewer, and an interactive
panel is available in a live Jupyter session. By default it runs the model
in-process through the same model service as the API, so it needs no server; it
can also call a running API.

All three clients render the same response, highlight the normalised text the
model actually scored, show direction by more than colour alone, and display
the same caution that the model assists triage and does not decide.

## 5.3 Hosting

The service runs on Render's free web-service tier. `render.yaml` declares the
service and `deploy/render/Dockerfile` builds it from a clone of the
repository: Python 3.11, CPU-only PyTorch and only the serving dependencies,
then the code, the configuration, the checkpoint, the vocabulary and the
threshold file. Data and notebooks never enter the image. A configuration
override sets the bind address and port, so no serving code differs between a
local run and the hosted one. Render rebuilds and redeploys on every push to
the main branch (Figure 11 shows the interactive API documentation of the
hosted service). A Hugging Face Docker Space configuration was prepared first
and is kept in `deploy/huggingface/`, but Docker Spaces on free hardware now
require a paid subscription.

The free instance has 512 MB of memory and a fraction of one CPU. Measured
under a 512 MB cap before deployment, the service used about 300 MB at rest and
peaked at about 400 MB over five consecutive long explanations, with no
out-of-memory failure. Latency is the main cost of free hosting: an
explanation takes about 50 seconds on the hosted instance against 3 to 5
seconds on a laptop, because LIME's 1 000 forward passes run on a shared CPU.
The number of perturbations was deliberately not reduced for the hosted copy,
so the hosted explanations are the ones analysed in section 3. The instance
sleeps after 15 idle minutes, and the first request after that takes about a
minute.

## 5.4 Security

A phishing-detection service is itself an attack surface, so the hosted
deployment is hardened (`src/phishlens/serving/security.py`):

- **Abuse limits.** Each client may send 30 scoring requests and 6
  explanations a minute, and the service accepts 30 explanations a minute
  across all clients, because each explanation consumes substantial CPU time.
  At most 16 connections are served concurrently. Requests over a limit
  receive HTTP 429 with a `Retry-After` header.
- **Client identification.** Behind Render's proxies the client address comes
  from the `X-Forwarded-For` header, which clients can also write. The service
  counts only the entries added by the trusted proxy hops, from the right, so
  a forged header cannot evade the limit. This was verified against the live
  service: 30 requests in a minute were accepted, the 31st refused, and
  requests carrying forged headers were still refused.
- **Response headers.** Every response carries `X-Content-Type-Options`,
  `X-Frame-Options`, `Referrer-Policy` and `Strict-Transport-Security`. The
  browser UI is served with a content security policy that allows only the
  site's own scripts, styles and images.
- **Input handling.** Email text is inserted into pages as text, never as
  markup, in all three clients, so pasted content cannot execute. Cross-origin
  requests are not enabled. Nothing submitted is stored.
- **Supply chain and secrets.** The container runs as a non-root user. The
  deployed image's packages were audited and have no known vulnerabilities;
  three development dependencies with published advisories (JupyterLab,
  urllib3 and Tornado) were upgraded. No credential exists in the repository
  or its history, and the Kaggle token used to download the data stays in the
  user's home directory.

## 5.5 Testing and reproducibility

The repository has 211 automated tests. Eighty-four of them are guards on the
results themselves: no content-hash group crosses two splits, the metrics
match values computed by hand, early stopping restores the best epoch, and
explanation offsets land on the right characters for both LIME and Integrated
Gradients (Figure 12). Further tests cover the API's validation and limits,
the security headers, all three clients, and the notebook audit, which checks
that every notebook is committed with its outputs and that only the evaluation
notebook reads the test set.

Every metric file records the commit that produced it, the seed and the hash of
the split configuration. The analysis notebooks were re-run at the final
version of the code; every reported metric and confusion matrix reproduced
exactly. The two notebooks that train models were not re-run, because
retraining is the step most sensitive to the software environment and the
committed model is the one every number in this report describes.

## 5.6 What the system demonstrates

The system turns the model into something an analyst can use in three
settings: in a browser with nothing installed, on a desktop, or inside a
notebook. It is honest about its own limits in the places a user sees them:
every verdict carries its probability, its threshold and its reasons, the
caution that a person decides is on every screen, and the About page and the
Model page state the model's provenance and known weaknesses. The deployment
also shows the practical cost of the model choices made in section 2: the
Text-CNN is light enough to serve from 512 MB of memory, while the
explanation method, not the model, sets the response time.

## References cited in this section

- Ribeiro, M.T., Singh, S. & Guestrin, C. 2016. "Why should I trust you?":
  explaining the predictions of any classifier. *Proceedings of KDD 2016*.
