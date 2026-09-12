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
# # 01 — Data audit
#
# Purpose: report per-corpus row counts, class balance, empty/corrupted rows,
# duplicate rate and text-length distributions for the six pooled Kaggle
# corpora. Produces the evidence that decides the question. Author:
# Colile Sibanda.

# %%
import os
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Notebooks run with the kernel's cwd set to notebooks/; load_config() reads
# a path relative to the repo root, so move up to it first.
_repo_root = Path.cwd()
while not (_repo_root / "pyproject.toml").exists():
    _repo_root = _repo_root.parent
os.chdir(_repo_root)

from phishlens.config.loader import load_config

config = load_config()
commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

print(f"Description: commit hash: {commit_hash}")
print("Description: config file: configs/default.yaml")
print(f"Description: seed: {config.seed}")
print(f"Description: run date: {datetime.now().date().isoformat()}")

# %% [markdown]
# ## Load each corpus and tag it with its source

# %%
raw_dir = config.paths.raw
corpora = {}
for file_name in config.data.corpora:
    path = f"{raw_dir}/{file_name}"
    df = pd.read_csv(path)
    corpus_name = file_name.rsplit(".", 1)[0]
    df["corpus"] = corpus_name
    corpora[corpus_name] = df

pooled = pd.concat(corpora.values(), ignore_index=True)
print(f"Description: corpora loaded: {len(corpora)}")
print(f"Description: pooled row count: {len(pooled)}")

# %% [markdown]
# ## Per-corpus row counts and class balance
#
# The positive label (1) mixes true phishing (Nazario, Nigerian Fraud) with
# bulk spam (CEAS-08, Enron, Ling, SpamAssassin). This split is exactly what
# decides the question below.

# %%
counts_table = (
    pooled.groupby("corpus")[config.data.label_column]
    .value_counts()
    .unstack(fill_value=0)
    .rename(columns={0: "legitimate (0)", 1: "phishing/spam (1)"})
)
counts_table["total"] = counts_table.sum(axis=1)
counts_table["positive_rate"] = (
    counts_table["phishing/spam (1)"] / counts_table["total"]
).round(3)
print(counts_table)

for corpus_name, row in counts_table.iterrows():
    print(
        f"Description: {corpus_name} rows: {int(row['total'])}, "
        f"positive rate: {row['positive_rate']}"
    )

# %%
fig, ax = plt.subplots(figsize=(8, 5))
counts_table[["legitimate (0)", "phishing/spam (1)"]].plot(
    kind="bar", stacked=True, ax=ax
)
ax.set_ylabel("row count")
ax.set_title("Per-corpus class balance")
plt.tight_layout()
plt.savefig(f"{config.paths.figures}/01_class_balance.png", dpi=config.evaluation.figure_dpi)
plt.show()

# %% [markdown]
# ## Empty and corrupted rows
#
# A row is corrupted if its combined subject+body text falls below
# `data.min_characters` after concatenation, or if the body is null.

# %%
text_columns = list(config.data.text_columns)
combined_text = pooled[text_columns].fillna("").agg(" ".join, axis=1).str.strip()
pooled["_combined_text"] = combined_text
pooled["_text_length"] = combined_text.str.len()

null_body = pooled["body"].isna().sum()
too_short = (pooled["_text_length"] < config.data.min_characters).sum()

print(f"Description: rows with null body: {null_body}")
print(
    f"Description: rows below min_characters ({config.data.min_characters}): {too_short}"
)
print(
    f"Description: rows corrupted or empty (union): "
    f"{(pooled['body'].isna() | (pooled['_text_length'] < config.data.min_characters)).sum()}"
)

# %% [markdown]
# ## Duplicate rate
#
# Exact duplicates on subject+body, pooled across all six corpora. This is
# the number that motivates the grouped content-hash split —
# a plain random split would let these duplicates land on both sides.

# %%
exact_duplicate_mask = pooled.duplicated(subset=text_columns, keep=False)
exact_duplicate_count = pooled.duplicated(subset=text_columns, keep="first").sum()
print(f"Description: rows sharing exact subject+body with another row: {exact_duplicate_mask.sum()}")
print(f"Description: exact duplicate rows (excluding first occurrence): {exact_duplicate_count}")
print(
    f"Description: duplicate rate: "
    f"{round(exact_duplicate_count / len(pooled), 4)}"
)

dup_by_corpus = (
    pooled[exact_duplicate_mask]
    .groupby("corpus")
    .size()
    .sort_values(ascending=False)
)
print(dup_by_corpus)

# %% [markdown]
# ## Text-length distribution

# %%
fig, ax = plt.subplots(figsize=(8, 5))
for corpus_name in corpora:
    subset = pooled.loc[pooled["corpus"] == corpus_name, "_text_length"]
    ax.hist(subset.clip(upper=5000), bins=50, alpha=0.5, label=corpus_name)
ax.set_xlabel("combined subject+body length (characters, clipped at 5000)")
ax.set_ylabel("row count")
ax.set_title("Text-length distribution by corpus")
ax.legend()
plt.tight_layout()
plt.savefig(f"{config.paths.figures}/01_text_length_distribution.png", dpi=config.evaluation.figure_dpi)
plt.show()

# %%
length_summary = pooled.groupby("corpus")["_text_length"].describe()[
    ["count", "mean", "50%", "min", "max"]
]
length_summary = length_summary.rename(columns={"50%": "median"})
print(length_summary)

# %% [markdown]
# ## Example emails from each class
#
# One phishing/spam (1) and one legitimate (0) example per corpus, truncated
# to 300 characters, so a reader can see the raw material behind the numbers
# above without opening the CSVs.

# %%
for corpus_name, df in corpora.items():
    print(f"--- {corpus_name} ---")
    for label_value in sorted(df[config.data.label_column].unique()):
        example = df.loc[df[config.data.label_column] == label_value].iloc[0]
        subject = str(example.get("subject", ""))[:100]
        body = str(example.get("body", ""))[:200].replace("\n", " ")
        print(f"  label={label_value} | subject: {subject!r}")
        print(f"  label={label_value} | body: {body!r}")

# %% [markdown]
# ## Deciding Q2
#
# **Question:** frame the task as "phishing/spam vs legitimate" (honest to
# the labels actually present in the pooled corpus), or restrict positives to
# Nazario + Nigerian Fraud only (honest to the project's name, "PhishLens",
# but confounds the positive class with exactly two source corpora)?
#
# **Evidence from this audit:**
# - Nazario and Nigerian Fraud are **100% positive** — neither corpus
#   contributes a single legitimate example. Restricting positives to these
#   two corpora would mean the negative class comes entirely from four other
#   corpora and the positive class from two, which is precisely the kind of
#   corpus/label confound this project's artefact guard exists to avoid
#.
# - CEAS-08, Enron, Ling and SpamAssassin all mix both classes, so the
#   "phishing/spam vs legitimate" framing draws its positive examples from
#   six different corpora and its negative examples from four — not a clean
#   split, but not a source-tied confound either.
# - Restricting positives to Nazario + Nigerian Fraud would also shrink the
#   positive class from tens of thousands of rows to ~4 900, discarding most
#   of the labelled phishing/spam signal in CEAS-08 and Enron.
#
# **Decision:** frame the task as **phishing/spam vs legitimate**, using the
# labels as provided across all six corpora. This is honest to what the
# labels actually encode, keeps the positive class from being tied to two
# in §4.1. The corpus probe and LOCO evaluation
# exist precisely to surface and report the corpus-confound risk this
# decision does not eliminate — restricting to Nazario + Nigerian Fraud
# would not remove that risk, it would just hide it inside a smaller,
# single-purpose positive class.
#

