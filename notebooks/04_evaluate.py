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
# # 04 — Evaluate on the test set (once)
#
# Purpose: the one and only place `test.csv` is opened. Evaluates M0 and
# M1 on test, reports the four required figures, the M0 vs M1 comparison
# at both the untuned threshold and each model's validation-selected
# threshold, the per-corpus breakdown for M1, and the false-positive
# review. Author: Colile Sibanda.
#
# **Test-set invariant:** no threshold is chosen here — `select_threshold`
# runs on `validation.csv` only, then is applied unchanged to test.

# %%
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Notebooks run with the kernel's cwd set to notebooks/; load_config() reads
# a path relative to the repo root, so move up to it first.
import os

_repo_root = Path.cwd()
while not (_repo_root / "pyproject.toml").exists():
    _repo_root = _repo_root.parent
os.chdir(_repo_root)

from phishlens.config.loader import load_config
from phishlens.config.seeding import set_global_seed
from phishlens.data.splitter import config_hash
from phishlens.evaluation.metrics import compute_metrics, select_threshold
from phishlens.evaluation.persistence import save_metrics
from phishlens.evaluation.plots import (
    plot_confusion_matrix,
    plot_loss_curve,
    plot_precision_recall,
    plot_roc,
)
from phishlens.evaluation.reporting import metrics_table, per_corpus_table
from phishlens.features.encoder import encode
from phishlens.features.vocabulary import Vocabulary
from phishlens.models.baseline import build_baseline
from phishlens.models.text_cnn import build_text_cnn
from phishlens.training.checkpoint import restore_best
from phishlens.training.loop import CHECKPOINT_FILE, METRIC_FILE

config = load_config()
set_global_seed(config.seed)
commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

print(f"Config seed: {config.seed}")
print(f"Commit hash: {commit_hash}")
print(f"Run date: {datetime.now().isoformat(timespec='seconds')}")

# %% [markdown]
# ## Load all three splits
#
# `test.csv` is opened here, once, per the invariant above.

# %%
processed_dir = Path(config.paths.processed)
train_df = pd.read_csv(processed_dir / "train.csv")
validation_df = pd.read_csv(processed_dir / "validation.csv")
test_df = pd.read_csv(processed_dir / "test.csv")

label_column = config.data.label_column

print(f"Train rows: {len(train_df)}")
print(f"Validation rows: {len(validation_df)}")
print(f"Test rows: {len(test_df)}")
print(f"Test positive rate: {test_df[label_column].mean():.4f}")

# %% [markdown]
# ## Rebuild M1 and restore the best checkpoint
#
# The vocabulary and weights come from `03_train_cnn.ipynb` — this notebook
# does not retrain.

# %%
vocabulary = Vocabulary.load(Path(config.paths.models) / "vocabulary.json")
model = build_text_cnn(vocab_size=vocabulary.size(), config=config)

checkpoint_path = Path(config.paths.models) / CHECKPOINT_FILE
restore_best(model, checkpoint_path)
model.eval()

print(f"Restored checkpoint: {checkpoint_path}")


def predict_m1(texts: list[str]) -> np.ndarray:
    """Encode texts and return M1 probabilities from the restored model.

    Inputs: texts — cleaned text strings from any split.
    Output: 1-D numpy array of sigmoid probabilities for the positive class.
    """
    sequences = encode(texts, vocabulary, config)
    with torch.no_grad():
        logits = model(torch.from_numpy(sequences))
    return torch.sigmoid(logits).numpy()


# %% [markdown]
# ## Refit M0 on train, per `02_baseline.ipynb`
#
# M0 has no persisted artefact from step 7, so it is refit here on the same
# training split for a fair test-set comparison.

# %%
m0_pipeline = build_baseline(config)
m0_pipeline.fit(train_df["text"].tolist(), train_df[label_column].to_numpy())

# %% [markdown]
# ## Score both models on validation and test

# %%
m0_val_probabilities = m0_pipeline.predict_proba(validation_df["text"].tolist())[:, 1]
m0_test_probabilities = m0_pipeline.predict_proba(test_df["text"].tolist())[:, 1]

m1_val_probabilities = predict_m1(validation_df["text"].tolist())
m1_test_probabilities = predict_m1(test_df["text"].tolist())

val_labels = validation_df[label_column].to_numpy()
test_labels = test_df[label_column].to_numpy()

# %% [markdown]
# ## Select thresholds on validation only
#
# `select_threshold` never sees test. The untuned 0.5 and the selected
# threshold are both reported below.

# %%
default_threshold = config.evaluation.decision_threshold
m0_selected_threshold = select_threshold(val_labels, m0_val_probabilities, objective="macro_f1")
m1_selected_threshold = select_threshold(val_labels, m1_val_probabilities, objective="macro_f1")

print(f"Untuned threshold: {default_threshold}")
print(f"M0 selected threshold (validation): {m0_selected_threshold:.4f}")
print(f"M1 selected threshold (validation): {m1_selected_threshold:.4f}")

# %% [markdown]
# ## Test-set metrics at both thresholds

# %%
m0_metrics_default = compute_metrics(test_labels, m0_test_probabilities, default_threshold)
m1_metrics_default = compute_metrics(test_labels, m1_test_probabilities, default_threshold)

m0_metrics_selected = compute_metrics(test_labels, m0_test_probabilities, m0_selected_threshold)
m1_metrics_selected = compute_metrics(test_labels, m1_test_probabilities, m1_selected_threshold)

# %% [markdown]
# ## M0 vs M1 comparison — untuned threshold (0.5)

# %%
metrics_table(
    {
        "M0 TF-IDF + LogReg": m0_metrics_default,
        "M1 Text-CNN": m1_metrics_default,
    }
)

# %% [markdown]
# ## M0 vs M1 comparison — validation-selected threshold

# %%
metrics_table(
    {
        "M0 TF-IDF + LogReg": m0_metrics_selected,
        "M1 Text-CNN": m1_metrics_selected,
    }
)

# %% [markdown]
# ## The four required figures (M1, test split, selected threshold)

# %%
plot_roc(test_labels, m1_test_probabilities, "M1 Text-CNN", "test").show()

# %%
plot_precision_recall(test_labels, m1_test_probabilities, "M1 Text-CNN", "test").show()

# %%
plot_confusion_matrix(m1_metrics_selected, "M1 Text-CNN", "test").show()

# %%
training_log_path = Path(config.paths.metrics) / METRIC_FILE
training_history = pd.read_csv(training_log_path).to_dict("records")
plot_loss_curve(training_history, "M1 Text-CNN").show()

# %% [markdown]
# ## Per-corpus breakdown for M1 (test split, selected threshold)

# %%
per_corpus_metrics = {}
for corpus, group in test_df.groupby("corpus"):
    group_probabilities = predict_m1(group["text"].tolist())
    per_corpus_metrics[corpus] = compute_metrics(
        group[label_column].to_numpy(), group_probabilities, m1_selected_threshold
    )

per_corpus_table(per_corpus_metrics)

# %% [markdown]
# **Reading Nazario and Nigerian_Fraud:** both corpora are 100% phishing in
# every split (see `docs/dataset-card.md` and the the build finding in
# `average_precision` are correctly NaN (undefined with a single class
# present, per `compute_metrics`'s documented behaviour) and `macro_f1`
# collapses to roughly half of the true positive-class F1, because macro
# averaging still divides by two classes even though the legitimate class
# has zero support here. Precision, recall and accuracy on the phishing
# class itself are the numbers that mean something for these two rows; they
# are near-perfect. This is a property of the corpus, not a model failure.

# %% [markdown]
# ## False-positive review (M1, test split, selected threshold)
#
# Legitimate emails (label 0) the model scored above the operating
# threshold — the qualitative evidence for the report's error analysis.

# %%
predictions = (m1_test_probabilities >= m1_selected_threshold).astype(int)
false_positive_mask = (test_labels == 0) & (predictions == 1)

false_positives = test_df.loc[false_positive_mask].copy()
false_positives["probability"] = m1_test_probabilities[false_positive_mask]
false_positives = false_positives.sort_values("probability", ascending=False)

print(f"False positives: {int(false_positive_mask.sum())} of {len(test_df)} test rows")
false_positives[["corpus", "probability", "text"]].head(10)

# %% [markdown]
# ## Save test-set metrics
#
# M0's test-set numbers are new — `m0_baseline.json` from step 7 is
# validation-only and must not be overwritten with test numbers.

# %%
run_context = {
    "seed": config.seed,
    "commit_hash": commit_hash,
    "config_hash": config_hash(config),
    "split_evaluated": "test",
    "train_rows": len(train_df),
    "validation_rows": len(validation_df),
    "test_rows": len(test_df),
}

save_metrics(
    m1_metrics_selected,
    Path(config.paths.metrics) / "m1_text_cnn.json",
    {**run_context, "model": "m1_text_cnn", "threshold_policy": "validation_selected_macro_f1"},
)
save_metrics(
    m0_metrics_selected,
    Path(config.paths.metrics) / "m0_test.json",
    {**run_context, "model": "m0_tfidf_logreg", "threshold_policy": "validation_selected_macro_f1"},
)

print("Metrics written to artifacts/metrics/m1_text_cnn.json and m0_test.json")
