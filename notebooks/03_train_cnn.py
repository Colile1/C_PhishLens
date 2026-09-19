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
# # 03 — Train M1 (Text-CNN)
#
# Purpose: build the training vocabulary, train the Text-CNN with the
# explicit patience-based early-stopping loop, and show the training curve
# and per-epoch metric table. Author: Colile Sibanda.
#
# **Test-set invariant:** this notebook touches `train.csv` and
# `validation.csv` only. `test.csv` is loaded in `04_evaluate.ipynb`, once.

# %%
import inspect
import os
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

# Notebooks run with the kernel's cwd set to notebooks/; load_config() reads
# a path relative to the repo root, so move up to it first.
_repo_root = Path.cwd()
while not (_repo_root / "pyproject.toml").exists():
    _repo_root = _repo_root.parent
os.chdir(_repo_root)

from phishlens.config.loader import load_config
from phishlens.config.seeding import set_global_seed
from phishlens.evaluation.plots import plot_loss_curve
from phishlens.features.encoder import encode
from phishlens.features.vocabulary import build_vocabulary
from phishlens.models.text_cnn import TextCNN, build_text_cnn, count_parameters
from phishlens.training.batching import make_dataset, make_loader
from phishlens.training.loop import train
from phishlens.training.weighting import positive_class_weight

config = load_config()
set_global_seed(config.seed)
commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

print(f"Config seed: {config.seed}")
print(f"Commit hash: {commit_hash}")
print(f"Run date: {datetime.now().isoformat(timespec='seconds')}")

# %% [markdown]
# ## Load the training and validation splits
#
# `test.csv` is never read here — see the invariant above.

# %%
processed_dir = Path(config.paths.processed)
train_df = pd.read_csv(processed_dir / "train.csv")
validation_df = pd.read_csv(processed_dir / "validation.csv")

label_column = config.data.label_column

print(f"Train rows: {len(train_df)}")
print(f"Validation rows: {len(validation_df)}")
print(f"Train positive rate: {train_df[label_column].mean():.4f}")
print(f"Validation positive rate: {validation_df[label_column].mean():.4f}")

# %% [markdown]
# ## Build the vocabulary
#
# Fitted on training text only — see `src/phishlens/features/vocabulary.py`.
# Saved so `04_evaluate.ipynb` and the serving layer encode with the exact
# same mapping.

# %%
vocabulary = build_vocabulary(train_df["text"].tolist(), config)
vocabulary_path = Path(config.paths.models) / "vocabulary.json"
vocabulary.save(vocabulary_path)

print(f"Vocabulary size: {vocabulary.size()}")
print(f"Vocabulary written to: {vocabulary_path}")

# %% [markdown]
# ## Encode text and build loaders

# %%
train_sequences = encode(train_df["text"].tolist(), vocabulary, config)
validation_sequences = encode(validation_df["text"].tolist(), vocabulary, config)

train_dataset = make_dataset(train_sequences, train_df[label_column].to_numpy())
validation_dataset = make_dataset(validation_sequences, validation_df[label_column].to_numpy())

train_loader = make_loader(train_dataset, config, shuffle=True)
validation_loader = make_loader(validation_dataset, config, shuffle=False)

positive_weight = positive_class_weight(train_df[label_column].to_numpy())
print(f"Positive-class weight (train only): {positive_weight.item():.4f}")

# %% [markdown]
# ## The model class
#
# Printed with `inspect.getsource` so the architecture is readable without
# leaving the notebook, per the notebook README and the question fallback.

# %%
print(inspect.getsource(TextCNN))

# %% [markdown]
# ## Build the model

# %%
model = build_text_cnn(vocab_size=vocabulary.size(), config=config)
print(f"Trainable parameters: {count_parameters(model):,}")
print(model)

# %% [markdown]
# ## The training loop
#
# Printed with `inspect.getsource` so the patience and best-weight-restore
# logic is readable without leaving the notebook.

# %%
print(inspect.getsource(train))

# %% [markdown]
# ## Train

# %%
result = train(
    model,
    train_loader,
    validation_loader,
    config,
    positive_weight=positive_weight,
    verbose=True,
)

print(f"Best epoch: {result.best_epoch}")
print(f"Best score ({config.training.early_stopping.metric}): {result.best_score:.4f}")
print(f"Epochs run: {result.epochs_run}")
print(f"Stopped early: {result.stopped_early}")
print(f"Checkpoint: {result.checkpoint_path}")

# %% [markdown]
# ## Training curve

# %%
figure = plot_loss_curve(result.history, "M1 Text-CNN")
figure.show()

# %% [markdown]
# ## Per-epoch metric table
#
# One row per epoch, all train/validation columns — the evidence for the
# per-iteration metric-logging marks. Backed by
# `artifacts/metrics/training_log.csv`, rewritten every epoch.

# %%
history_df = pd.DataFrame(result.history)
history_df
