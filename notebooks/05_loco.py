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
# # 05 : Leave-one-corpus-out evaluation
#
# Purpose: the honest generalisation check the artefact guard exists to
# produce. Six folds, one per
# source corpus: train M1 on the other five corpora, test on the held-out
# one, with content-hash groups shared between the two sides removed from
# train first. Reports the six LOCO F1 scores beside the headline
# random-split F1 and the corpus-probe accuracy in one place. Author:
# Colile Sibanda.
#
# **Regimen:** every fold trains with the identical
# config as `03_train_cnn.ipynb` : same architecture, batch size, learning
# rate, patience-3 early stopping on validation macro-F1, best-weight
# restore. No shortcut, so the six numbers are comparable to the headline
# run. Each fold gets its own validation carve-out (a grouped, stratified
# 85/15 split of the fold's train-side groups, reusing
# `phishlens.data.splitter.make_split`) so early stopping has a signal
# independent of the held-out corpus.
#
# **This notebook does not touch `data/processed/test.csv`.** LOCO folds are
# built from the union of all three existing splits (train + validation +
# test), which is a different resampling than the headline train/test split,
# not a re-use of the reserved test set for model selection : every LOCO
# fold's own test side is scored only after that fold's own training is
# finished, exactly once, the same discipline `04_evaluate.ipynb` uses.

# %%
import dataclasses
import inspect
import os
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import psutil
import torch

# Notebooks run with the kernel's cwd set to notebooks/; load_config() reads
# a path relative to the repo root, so move up to it first.
_repo_root = Path.cwd()
while not (_repo_root / "pyproject.toml").exists():
    _repo_root = _repo_root.parent
os.chdir(_repo_root)

from phishlens.config.loader import load_config
from phishlens.config.seeding import set_global_seed
from phishlens.data.loco import loco_report, make_loco_index
from phishlens.data.splitter import make_split
from phishlens.evaluation.metrics import compute_metrics, select_threshold
from phishlens.evaluation.persistence import commit_hash, save_metrics
from phishlens.evaluation.reporting import per_corpus_table
from phishlens.features.encoder import encode
from phishlens.features.vocabulary import build_vocabulary
from phishlens.models.text_cnn import build_text_cnn
from phishlens.training.batching import make_dataset, make_loader
from phishlens.training.loop import train
from phishlens.training.weighting import positive_class_weight

config = load_config()
set_global_seed(config.seed)
run_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

print(f"Config seed: {config.seed}")
print(f"Commit hash: {run_commit}")
print(f"Run date: {datetime.now().isoformat(timespec='seconds')}")
print(f"Available RAM: {psutil.virtual_memory().available / 1e9:.2f} GB")

# %% [markdown]
# ## Load the pooled frame
#
# All three splits carry `content_hash` and `corpus` already (written by
# `write_split`), so the pool is reassembled without
# re-reading raw data. Concatenating train + validation + test is not a
# reuse of the reserved test set for a modelling decision : LOCO folds each
# split are their own train/test resampling, described in the module
# docstring above.

# %%
processed_dir = Path(config.paths.processed)
label_column = config.data.label_column

pooled = pd.concat(
    [
        pd.read_csv(processed_dir / "train.csv"),
        pd.read_csv(processed_dir / "validation.csv"),
        pd.read_csv(processed_dir / "test.csv"),
    ],
    ignore_index=True,
)

print(f"Pooled rows: {len(pooled)}")
print(f"Pooled positive rate: {pooled[label_column].mean():.4f}")
print(f"Corpora: {sorted(pooled['corpus'].unique())}")

# %% [markdown]
# ## Build the LOCO fold index
#
# `make_loco_index` (unit-tested in
# `tests/unit/test_data.py::TestLoco`) returns one fold per corpus: the
# held-out corpus's groups on the test side, every other corpus's groups on
# the train side with any group also present in the held-out corpus removed
# first : the step that keeps a fold from measuring the leakage the random
# split already guards against.

# %%
folds = make_loco_index(pooled)
report = loco_report(pooled, folds, label_column=label_column)
report

# %% [markdown]
# ## Per-fold training and evaluation
#
# One full training run per corpus. Each fold:
#
# 1. Carves the fold's train-side groups into an 85/15 train/validation
#    split with `make_split` : the same grouped, stratified splitter used
#    for the headline split, reused here rather than reimplemented, so LOCO
#    inherits the same leakage guarantee within each fold.
# 2. Builds a vocabulary on that fold's train rows only.
# 3. Trains M1 with the identical config as the headline run :
#    patience 3 on validation macro-F1, best-weight restore.
# 4. Scores the restored model on the held-out corpus, at both the untuned
#    0.5 threshold and a threshold selected on the fold's own validation
#    split (the same validation-only discipline as `04_evaluate.ipynb`).
#
# Checkpoints and per-fold metric CSVs are written under
# `artifacts/models/loco/<corpus>/` and `artifacts/metrics/loco/<corpus>/`
# via a per-fold config (`dataclasses.replace` on `paths`), so no fold
# overwrites the headline M1 checkpoint.

# %%
print(inspect.getsource(train))

# %%
import gc

import numpy as np

# Held-out corpora are large (CEAS_08 is ~39k emails). One forward pass over a
# whole corpus asked for a single 6.5 GB activation tensor and crashed the
# 2026-09-26 re-run with "not enough memory". Predicting in fixed-size batches
# caps peak memory at one batch. Outputs are unchanged: the model is in eval
# mode (dropout off) and has no batch-dependent layers, so each email's
# probability does not depend on which other emails share its batch.
INFERENCE_BATCH_SIZE = 512


def predict_in_batches(model, sequences):
    """Return sigmoid probabilities for encoded sequences, one batch at a time.

    Inputs: model : a TextCNN already in eval mode; sequences : 2-D int array
    of encoded token ids.
    Output: 1-D numpy array of P(phishing), one per row of `sequences`.
    """
    chunks = []
    with torch.no_grad():
        for start in range(0, len(sequences), INFERENCE_BATCH_SIZE):
            batch = torch.from_numpy(sequences[start:start + INFERENCE_BATCH_SIZE])
            chunks.append(torch.sigmoid(model(batch)).numpy())
    return np.concatenate(chunks)

fold_results: dict[str, dict] = {}
fold_metrics_selected = {}
fold_metrics_default = {}

# Each iteration builds a fresh vocabulary, encoded tensor datasets, loaders
# and a TextCNN. In one long-lived kernel these do not free themselves
# between folds unless explicitly deleted, so six folds accumulate six
# folds' worth of tensors : this crashed the first run attempt with an OOM
# partway through fold 3 (Ling), confirmed by `psutil.virtual_memory()`
# reading under 50MB available immediately before the kernel died. Every
# large per-fold object is deleted and gc.collect() is called at the end of
# each iteration so peak memory reflects one fold, not the running total.
for corpus in sorted(folds):
    print(f"\n{'=' * 60}")
    print(f"Fold: held out = {corpus}")
    print(f"{'=' * 60}")
    print(f"Available RAM before fold: {psutil.virtual_memory().available / 1e9:.2f} GB")

    fold = folds[corpus]
    fold_train_side = pooled[pooled["content_hash"].isin(fold.train)].reset_index(drop=True)
    fold_test_side = pooled[pooled["content_hash"].isin(fold.test)].reset_index(drop=True)

    # Per-fold paths so this fold's checkpoint and metric log cannot collide
    # with another fold's or with the headline run's.
    fold_paths = dataclasses.replace(
        config.paths,
        models=str(Path(config.paths.models) / "loco" / corpus),
        metrics=str(Path(config.paths.metrics) / "loco" / corpus),
    )
    # 85/15 train/validation split of the fold's train-side groups, reusing
    # the project splitter. `test` is set to 0.0 so every non-validation row
    # lands in `train` : the held-out corpus is not part of this call at all.
    fold_split_config = dataclasses.replace(
        config,
        paths=fold_paths,
        split=dataclasses.replace(config.split, train=0.85, validation=0.15, test=0.0),
    )
    inner_split = make_split(fold_train_side, fold_split_config)
    inner_labelled = inner_split.apply(fold_train_side)

    inner_train_df = inner_labelled[inner_labelled["split"] == "train"].reset_index(drop=True)
    inner_val_df = inner_labelled[inner_labelled["split"] == "validation"].reset_index(drop=True)

    print(f"Fold train rows: {len(inner_train_df)}, validation rows: {len(inner_val_df)}, "
          f"held-out test rows: {len(fold_test_side)}")

    fold_vocabulary = build_vocabulary(inner_train_df["text"].tolist(), fold_split_config)

    train_sequences = encode(inner_train_df["text"].tolist(), fold_vocabulary, fold_split_config)
    val_sequences = encode(inner_val_df["text"].tolist(), fold_vocabulary, fold_split_config)
    test_sequences = encode(fold_test_side["text"].tolist(), fold_vocabulary, fold_split_config)

    train_dataset = make_dataset(train_sequences, inner_train_df[label_column].to_numpy())
    val_dataset = make_dataset(val_sequences, inner_val_df[label_column].to_numpy())

    train_loader = make_loader(train_dataset, fold_split_config, shuffle=True)
    val_loader = make_loader(val_dataset, fold_split_config, shuffle=False)

    positive_weight = positive_class_weight(inner_train_df[label_column].to_numpy())

    model = build_text_cnn(vocab_size=fold_vocabulary.size(), config=fold_split_config)

    result = train(
        model,
        train_loader,
        val_loader,
        fold_split_config,
        positive_weight=positive_weight,
        verbose=True,
    )

    print(f"Best epoch: {result.best_epoch}, best {config.training.early_stopping.metric}: "
          f"{result.best_score:.4f}, stopped early: {result.stopped_early}")

    model.eval()
    val_probabilities = predict_in_batches(model, val_sequences)
    test_probabilities = predict_in_batches(model, test_sequences)

    val_labels = inner_val_df[label_column].to_numpy()
    test_labels = fold_test_side[label_column].to_numpy()

    default_threshold = config.evaluation.decision_threshold
    # Nazario and Nigerian_Fraud are 100% positive in every split (see
    # docs/dataset-card.md) : the fold's own validation carve-out inherits that, so
    # select_threshold has no negative class to define an operating point
    # against. Fall back to the untuned threshold for those folds and say so.
    if val_labels.min() == val_labels.max():
        selected_threshold = default_threshold
        print(f"Validation carve-out is single-class for {corpus}; "
              f"using the untuned threshold {default_threshold} instead of a selected one.")
    else:
        selected_threshold = select_threshold(val_labels, val_probabilities, objective="macro_f1")

    metrics_default = compute_metrics(test_labels, test_probabilities, default_threshold)
    metrics_selected = compute_metrics(test_labels, test_probabilities, selected_threshold)

    fold_metrics_default[corpus] = metrics_default
    fold_metrics_selected[corpus] = metrics_selected
    fold_results[corpus] = {
        "best_epoch": result.best_epoch,
        "best_val_score": result.best_score,
        "epochs_run": result.epochs_run,
        "stopped_early": result.stopped_early,
        "selected_threshold": selected_threshold,
        "train_rows": len(inner_train_df),
        "validation_rows": len(inner_val_df),
        "test_rows": len(fold_test_side),
    }

    run_context = {
        "seed": config.seed,
        "commit_hash": run_commit,
        "held_out_corpus": corpus,
        "split_evaluated": "loco_held_out",
        "threshold_policy": "validation_selected_macro_f1"
        if val_labels.min() != val_labels.max()
        else "untuned_0.5_fallback_single_class_validation",
        "train_rows": len(inner_train_df),
        "validation_rows": len(inner_val_df),
        "test_rows": len(fold_test_side),
        "best_epoch": result.best_epoch,
        "epochs_run": result.epochs_run,
        "stopped_early": result.stopped_early,
    }
    save_metrics(
        metrics_selected,
        Path(config.paths.metrics) / "loco" / f"{corpus}.json",
        run_context,
    )

    # Free this fold's large objects before the next iteration builds its
    # own : see the note above the loop. Local variables named here are
    # every array/tensor/model/dataset scoped to one fold.
    del (
        model, result,
        fold_train_side, fold_test_side, inner_labelled,
        inner_train_df, inner_val_df,
        fold_vocabulary,
        train_sequences, val_sequences, test_sequences,
        train_dataset, val_dataset, train_loader, val_loader,
        positive_weight,
        val_probabilities, test_probabilities,
        val_labels, test_labels,
    )
    gc.collect()
    print(f"Available RAM after fold cleanup: {psutil.virtual_memory().available / 1e9:.2f} GB")

print("\nAll six folds complete.")

# %% [markdown]
# ## LOCO results : untuned threshold (0.5)

# %%
per_corpus_table(fold_metrics_default)

# %% [markdown]
# ## LOCO results : validation-selected threshold
#
# Nazario and Nigerian_Fraud fall back to the untuned threshold (see the
# per-fold print output above) because their validation carve-outs are
# single-class, the same property noted for their test-split rows in
# `04_evaluate.ipynb`.

# %%
loco_table = per_corpus_table(fold_metrics_selected)
loco_table

# %% [markdown]
# ## Fold sizes and run details

# %%
fold_details = pd.DataFrame(fold_results).T
fold_details.index.name = "held_out_corpus"
fold_details

# %% [markdown]
# ## The headline comparison: LOCO F1 vs random-split F1 vs corpus-probe accuracy
#
# The three numbers that tell the honest generalisation story together:
# the corpus probe shows how
# much of the pool's separability is corpus recognition rather than
# phishing signal; the random-split F1 is the headline number a reader sees
# first; the LOCO F1 is what M1 scores on a corpus it never saw during
# training, which is the more honest generalisation estimate.

# %%
import json

headline_path = Path(config.paths.metrics) / "m1_text_cnn.json"
probe_path = Path(config.paths.metrics) / "corpus_probe.json"

headline_macro_f1 = None
probe_accuracy = None
if headline_path.is_file():
    headline_macro_f1 = json.loads(headline_path.read_text())["metrics"]["macro_f1"]
if probe_path.is_file():
    probe_accuracy = json.loads(probe_path.read_text())["accuracy"]

# LOCO's macro_f1 for Nazario/Nigerian_Fraud is not comparable (single-class
# test side, see 04_evaluate.ipynb's reading note) : the mean below is taken
# over the four corpora where macro_f1 is well-defined, and stated as such.
two_class_corpora = [c for c in loco_table.index if c not in ("Nazario", "Nigerian_Fraud")]
loco_mean_macro_f1 = loco_table.loc[two_class_corpora, "macro_f1"].mean()

print(f"Corpus probe accuracy: {probe_accuracy:.4f}" if probe_accuracy is not None else "Corpus probe metrics not found.")
print(f"Headline random-split M1 macro-F1 (test split): {headline_macro_f1:.4f}" if headline_macro_f1 is not None else "Headline M1 metrics not found.")
print(f"LOCO mean macro-F1 across the four two-class corpora "
      f"({', '.join(two_class_corpora)}): {loco_mean_macro_f1:.4f}")
if headline_macro_f1 is not None:
    print(f"Gap (headline − LOCO mean): {headline_macro_f1 - loco_mean_macro_f1:.4f}")

# %% [markdown]
# **The one-sentence generalisation story for the report (fill in after the
# run above executes and the printed numbers are known):** M1's headline
# random-split macro-F1 of `{headline_macro_f1}` drops to a LOCO mean of
# `{loco_mean_macro_f1}` across the four two-class held-out corpora, a gap
# of `{headline − LOCO mean}` : measured against a corpus-probe accuracy of
# `{probe_accuracy}`, this quantifies how much of the headline
# number is genuine phishing signal that transfers to an unseen corpus
# versus corpus-specific artefact the random split lets the model exploit.
