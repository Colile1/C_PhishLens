# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 07 : Analyst UI in a notebook
#
# Purpose: use PhishLens from a notebook page. Paste an email, get the verdict,
# the probability against the decision threshold, and the words that pushed the
# model each way. Author: Colile Sibanda.
#
# **Two ways to use this page.**
#
# - **Read it**: the worked examples below were run and committed with their
#   outputs, so they display in any notebook viewer, GitHub included.
# - **Use it**: open it in Jupyter (`jupyter lab notebooks/`), run the setup cell,
#   then edit the text in *Analyse your own email*, or use the interactive panel at
#   the end.
#
# The model runs inside this notebook through the same `ModelService` the API
# serves from, so no server and no network are needed, and a verdict here is the
# verdict the API, the desktop app and the browser UI would give. The client code
# is in `src/notebook_ui/`.
#
# **Reading a result.** The verdict comes first. In the text below it, words in
# **bold** pushed towards phishing and words in *italic* towards legitimate;
# darker shading means a larger weight. The text is shown as the model saw it:
# lower case, with links, addresses and numbers replaced by `<url>`, `<email>` and
# `<num>`. The weights explain this one prediction only (LIME), and the model
# assists triage: an analyst still decides.

# %%
import os
import subprocess
from datetime import datetime
from pathlib import Path

# Notebooks run with the kernel's cwd set to notebooks/; the model files are
# found relative to the repository root, so move up to it first.
_repo_root = Path.cwd()
while not (_repo_root / "pyproject.toml").exists():
    _repo_root = _repo_root.parent
os.chdir(_repo_root)

from notebook_ui import ApiBackend, LocalBackend, analyse, launch

backend = LocalBackend()
run_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

print(f"Commit hash: {run_commit}")
print(f"Run date: {datetime.now().isoformat(timespec='seconds')}")
print("Backend: in-process model (LocalBackend)")

# %% [markdown]
# ## Example: a phishing email
#
# A typical credential lure: urgency, a threat of account closure, and a link.

# %%
PHISHING_EXAMPLE = """URGENT: your account has been suspended

Verify your password immediately at http://secure-update.example/login or your
account will be closed within 24 hours. Click here to confirm your identity now."""

analyse(PHISHING_EXAMPLE, backend)

# %% [markdown]
# ## Example: a legitimate email
#
# An ordinary internal message about a meeting.

# %%
LEGITIMATE_EXAMPLE = """Subject: minutes from tuesday meeting

Hi all, attached are the minutes from the project meeting. Please review the
action items before next week and let me know if I missed anything. Thanks, Sam"""

analyse(LEGITIMATE_EXAMPLE, backend)

# %% [markdown]
# ## Analyse your own email
#
# Replace the text between the triple quotes with the subject and body of any
# email (leave the headers out: the model does not use them), then run the cell.
# The committed output shows a parcel-delivery lure as an example.

# %%
MY_EMAIL = """Your parcel could not be delivered

We attempted to deliver your package today but no one was available. To arrange
a new delivery, confirm your address and pay the R25 redelivery fee at
http://parcel-redelivery.example/track within 48 hours, or the package will be
returned to the sender."""

analyse(MY_EMAIL, backend)

# %% [markdown]
# ## Interactive panel
#
# With a running Jupyter kernel, this cell shows a text box and an **Analyse
# email** button; the result appears under it. A static viewer such as GitHub
# cannot run widgets and shows only a one-line placeholder here, which is why the
# examples above use `analyse(...)` directly.

# %%
launch(backend)

# %% [markdown]
# ## Using a running API instead
#
# The same page can call the PhishLens API rather than loading the model locally,
# for example the hosted demo. Explanations take up to a minute on the free host.
#
# ```python
# hosted = ApiBackend("https://phishlens-vdc9.onrender.com")
# analyse(MY_EMAIL, hosted)
# launch(hosted)
# ```
#
# Or a local API started with `python run_api.py`:
# `ApiBackend("http://127.0.0.1:8000")`.
