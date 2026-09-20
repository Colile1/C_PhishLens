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
# # 06 : Explainability : LIME token attributions for M0 and M1
#
# Purpose: turn each model's verdict into evidence an analyst can read. For
# the same emails, explain M0 (TF-IDF + Logistic Regression) and M1
# (Text-CNN) with LIME, compare what each one keys on, and measure how far
# an explanation moves when only LIME's random seed changes. This is build
# step 12 and the XAI half of the extra features. Author: Colile Sibanda.
#
# **Sign convention, stated once and followed everywhere** (see
# `src/phishlens/explain/README.md`): a **positive** weight pushes the
# prediction **towards phishing** (label 1); a negative weight pushes it
# towards legitimate (label 0).
#
# **What is explained.** LIME is handed the *normalised*
# text, not the raw email, and the project tokeniser is handed to LIME as its
# `split_expression`. The normalised text is what the model actually sees:
# it has been lowercased and has had URLs, email addresses and number runs
# replaced by `<url>`, `<email>` and `<num>`. Attributing a weight to a raw
# URL span would claim an attribution the model never made : it saw `<url>`,
# one token, not a domain. Every character offset below therefore indexes
# into the normalised string, which each `Explanation` carries with it.
#
# **An explanation is local.** These weights describe *this* prediction in
# the neighbourhood of *this* email, fitted by a surrogate linear model on
# perturbations of it. They are not a statement about the model in general
# (Ribeiro et al., 2016). The report and the desktop app both say so.
#
# **This notebook does not touch `data/processed/test.csv`.** Per
# `notebooks/README.md` the test split is loaded in `04_evaluate.ipynb`
# only, exactly once. Every email explained here comes from the validation
# split, including the false-positive review : validation false positives
# make the same qualitative point without spending the test set a second
# time.

# %%
import inspect
import os
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

# Notebooks run with the kernel's cwd set to notebooks/; load_config() reads
# a path relative to the repo root, so move up to it first.
_repo_root = Path.cwd()
while not (_repo_root / "pyproject.toml").exists():
    _repo_root = _repo_root.parent
os.chdir(_repo_root)

from phishlens.config.loader import load_config
from phishlens.config.seeding import set_global_seed
from phishlens.evaluation.metrics import compute_metrics, select_threshold
from phishlens.explain.attributions import align_to_tokens, top_tokens
from phishlens.explain.lime_explainer import explain_text
from phishlens.explain.predict_fn import (
    make_baseline_predict_fn,
    make_text_cnn_predict_fn,
)
from phishlens.explain.stability import stability_report
from phishlens.features.normalise import normalise_text
from phishlens.features.vocabulary import Vocabulary
from phishlens.models.baseline import build_baseline
from phishlens.models.text_cnn import build_text_cnn
from phishlens.training.checkpoint import restore_best

config = load_config()
set_global_seed(config.seed)
run_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

print(f"Config seed: {config.seed}")
print(f"Commit hash: {run_commit}")
print(f"Run date: {datetime.now().isoformat(timespec='seconds')}")
print(f"Explain method: {config.explain.method}")
print(f"Tokens per explanation: {config.explain.num_features}")
print(f"LIME perturbations per explanation: {config.explain.num_samples}")
print(f"Stability seeds: {list(config.explain.stability_seeds)}")

# %% [markdown]
# ## The explanation path, printed in full
#
# Per `notebooks/README.md`, the logic a reader needs to read is printed
# into the notebook with `inspect.getsource` rather than only imported. The
# two functions below are the whole of the build correctness claim:
# `align_to_tokens` is the off-by-one guard, and `explain_text` is the LIME
# wrapper that decides what string is explained.
#
# Read `align_to_tokens` for the direction of the loop. It walks the
# **text's own tokens** and looks each one up in LIME's weight map, rather
# than walking the weight map and searching the text. That direction is what
# makes a misalignment impossible: the output is driven by the input, so a
# token can never be reported at another token's offsets, and a weight for a
# token absent from the text is simply never emitted.

# %%
print(inspect.getsource(align_to_tokens))

# %%
print(inspect.getsource(explain_text))

# %% [markdown]
# ## Load the splits and both models
#
# M1 is the checkpoint with the best epoch's weights
# restored : the same weights `04_evaluate.ipynb` reported test numbers for.
# M0 is refit here on the training split, exactly as `04_evaluate.ipynb`
# fits it, because the baseline pipeline is not persisted to disk.
#
# Both are wrapped by `explain/predict_fn.py` into the *same* two-column
# probability contract LIME expects. That wrapper is the same code path the
# `/explain` endpoint will call : `explain/README.md`
# requires that the function handed to LIME is the deployed path, not a
# reimplementation of it, since an explanation of a different pipeline is
# not an explanation.

# %%
processed_dir = Path(config.paths.processed)
label_column = config.data.label_column

train_df = pd.read_csv(processed_dir / "train.csv")
validation_df = pd.read_csv(processed_dir / "validation.csv")

print(f"Train rows: {len(train_df)}")
print(f"Validation rows: {len(validation_df)}")
print(f"Validation positive rate: {validation_df[label_column].mean():.4f}")

# %%
m0_pipeline = build_baseline(config)
m0_pipeline.fit(train_df["text"].tolist(), train_df[label_column].to_numpy())
m0_predict = make_baseline_predict_fn(m0_pipeline)

print(f"M0 fitted on {len(train_df)} training rows")
print(f"M0 classes: {list(m0_pipeline.classes_)}")

# %%
vocabulary = Vocabulary.load(Path(config.paths.models) / "vocabulary.json")
m1_model = build_text_cnn(vocabulary.size(), config)
restore_best(m1_model, Path(config.serving.model_file))
m1_predict = make_text_cnn_predict_fn(m1_model, vocabulary, config)

print(f"Vocabulary size: {vocabulary.size()}")
print(f"M1 checkpoint: {config.serving.model_file}")
print(f"M1 training mode (must be False): {m1_model.training}")

# %% [markdown]
# ## Choose the emails to explain
#
# Four validation emails, chosen to cover the cases an analyst actually
# meets rather than only the easy one:
#
# 1. a confidently-flagged phishing email,
# 2. a confidently-passed legitimate email,
# 3. a **false positive** : legitimate, but M1 flagged it (the review
#    `notebooks/README.md` asks this notebook for),
# 4. a **borderline** email, whose probability sits nearest the decision
#    threshold and where an explanation is worth the most.
#
# Selection uses M1's validation probabilities and the threshold selected on
# validation, the same the design discipline as `04_evaluate.ipynb`. Nothing here
# touches the test split.

# %%
validation_texts = validation_df["text"].tolist()
validation_labels = validation_df[label_column].to_numpy()

m1_validation_probabilities = m1_predict(validation_texts)[:, 1]
selected_threshold = select_threshold(
    validation_labels, m1_validation_probabilities, objective="macro_f1"
)

validation_scored = validation_df.assign(probability=m1_validation_probabilities)
validation_scored["predicted"] = (
    validation_scored["probability"] >= selected_threshold
).astype(int)

print(f"M1 selected threshold (validation): {selected_threshold:.4f}")
print(
    "M1 validation metrics: "
    f"{compute_metrics(validation_labels, m1_validation_probabilities, selected_threshold).as_dict()}"
)

# %%
confident_phish = validation_scored[
    (validation_scored[label_column] == 1) & (validation_scored["predicted"] == 1)
].nlargest(1, "probability")

confident_legitimate = validation_scored[
    (validation_scored[label_column] == 0) & (validation_scored["predicted"] == 0)
].nsmallest(1, "probability")

false_positives = validation_scored[
    (validation_scored[label_column] == 0) & (validation_scored["predicted"] == 1)
].nlargest(1, "probability")

borderline = validation_scored.assign(
    distance=(validation_scored["probability"] - selected_threshold).abs()
).nsmallest(1, "distance")

cases: list[tuple[str, pd.Series]] = []
for name, frame in [
    ("Confident phishing (true positive)", confident_phish),
    ("Confident legitimate (true negative)", confident_legitimate),
    ("False positive : legitimate, flagged", false_positives),
    ("Borderline : nearest the threshold", borderline),
]:
    if frame.empty:
        print(f"No validation example available for: {name}")
        continue
    cases.append((name, frame.iloc[0]))

print(f"\nFalse positives in validation: {len(validation_scored[(validation_scored[label_column] == 0) & (validation_scored['predicted'] == 1)])}")
print(f"Cases selected for explanation: {len(cases)}")
for name, row in cases:
    print(f"  {name}: corpus={row['corpus']}, label={row[label_column]}, P(phishing)={row['probability']:.4f}")


# %% [markdown]
# ## A readable rendering of one explanation
#
# `render_explanation` is presentation only : it prints the top tokens with
# their signed weights and marks the direction each one pushed. The
# attribution itself is computed in `src/phishlens/explain/`, so this cell
# adds no logic the tests do not already cover.

# %%
def render_explanation(title: str, explanation, top_n: int = 10) -> None:
    """Print one explanation's top tokens with signed weights and offsets.

    Inputs: title : heading for this block; explanation : an Explanation from
    explain_text; top_n : how many tokens to show.
    Output: none; prints to the notebook.
    """
    print(f"{title}")
    print(f"P(phishing): {explanation.probability:.4f}")
    print(f"Seed: {explanation.seed}")
    print(f"Tokens in email: {len(explanation.tokens)}")
    print(f"{'token':>18}  {'weight':>9}  direction   offsets")
    for entry in explanation.top[:top_n]:
        direction = "phishing" if entry.pushes_towards_phishing() else "legitimate"
        print(
            f"{entry.token!r:>18}  {entry.weight:+9.4f}  {direction:>10}   "
            f"[{entry.start}:{entry.end}]"
        )


def verify_offsets(explanation) -> bool:
    """Re-check in the notebook that every offset slices back to its token.

    The same invariant `tests/critical/test_attribution_alignment.py` proves,
    asserted again here on real emails so a reader sees it hold on the actual
    data rather than only in a test file.

    Inputs: explanation : an Explanation from explain_text.
    Output: True if every offset pair reproduces its own token.
    """
    return all(
        explanation.text[entry.start : entry.end] == entry.token
        for entry in explanation.tokens
    )


# %% [markdown]
# ## M0 against M1 on the same emails
#
# LIME is model-agnostic, which is exactly why it was chosen as the primary
# method: the two models are explained through
# the same interface, so the token lists below are directly comparable.
#
# M0 is a linear model over TF-IDF n-grams, so its explanation should track
# its own coefficients closely. M1 pools over convolutional phrase
# detectors, so it has the *option* of keying on multi-token lures. Where
# the two disagree is the interesting part for report section 3.3.

# %%
explanations: dict[str, dict[str, object]] = {}

for name, row in cases:
    email_text = row["text"]
    m0_explanation = explain_text(email_text, m0_predict, config)
    m1_explanation = explain_text(email_text, m1_predict, config)
    explanations[name] = {
        "row": row,
        "m0": m0_explanation,
        "m1": m1_explanation,
    }

    print("=" * 78)
    print(f"{name}")
    print(f"Corpus: {row['corpus']}   True label: {row[label_column]}")
    print("=" * 78)
    render_explanation("--- M0 TF-IDF + Logistic Regression ---", m0_explanation)
    print()
    render_explanation("--- M1 Text-CNN ---", m1_explanation)
    print()
    print(f"M0 offsets verified: {verify_offsets(m0_explanation)}")
    print(f"M1 offsets verified: {verify_offsets(m1_explanation)}")
    print()

# %% [markdown]
# ## Agreement between the two models' explanations
#
# The overlap of the two top-k token sets, per email. A high overlap says
# the two models reach the same verdict for the same stated reasons; a low
# overlap with the same verdict is the more uncomfortable result and the one
# worth writing about : two models agreeing by coincidence rather than by
# reading the same evidence.

# %%
agreement_rows = []
for name, bundle in explanations.items():
    m0_tokens = bundle["m0"].top_token_set()
    m1_tokens = bundle["m1"].top_token_set()
    union = m0_tokens | m1_tokens
    shared = m0_tokens & m1_tokens
    agreement_rows.append(
        {
            "case": name,
            "M0 P(phishing)": round(bundle["m0"].probability, 4),
            "M1 P(phishing)": round(bundle["m1"].probability, 4),
            "shared tokens": len(shared),
            "jaccard": round(len(shared) / len(union), 4) if union else 1.0,
            "shared": ", ".join(sorted(shared)[:8]),
        }
    )

agreement_table = pd.DataFrame(agreement_rows)
agreement_table

# %% [markdown]
# ## Stability across seeds
#
# LIME fits its surrogate to a random sample of `num_samples` perturbations,
# so the reported top-k token set is a random variable. Explaining the same
# email under each seed in `config.explain.stability_seeds` and comparing the
# results measures how much of an explanation is signal and how much is
# sampling noise.
#
# **The probability spread is the control.** Both models are deterministic at
# inference : M1 is in `eval()` mode with gradients disabled : so
# `P(phishing)` must be *identical* across seeds. Only LIME's sampling
# changed. A non-zero spread would mean something else is non-deterministic
# in the predict path, so it is reported rather than assumed.
#
# A low Jaccard overlap here is a **finding for the report, not a defect to
# hide** (`explain/README.md`): it would mean explanations at this
# `num_samples` are not reproducible, which is a stated limitation of the
# method and a caveat the desktop app's output inherits.

# %%
stability_rows = []
for name, bundle in explanations.items():
    email_text = bundle["row"]["text"]
    for model_name, predict in [("M0", m0_predict), ("M1", m1_predict)]:
        report = stability_report(email_text, predict, config)
        stability_rows.append(
            {
                "case": name,
                "model": model_name,
                "seeds": ", ".join(str(seed) for seed in report.seeds),
                "jaccard": round(report.jaccard, 4),
                "agreed tokens": len(report.agreed_tokens),
                "P(phishing) spread": report.probability_spread,
            }
        )
        print(f"{model_name} : {name}")
        print(f"  Jaccard overlap of top-{config.explain.num_features}: {report.jaccard:.4f}")
        print(f"  P(phishing) spread across seeds: {report.probability_spread:.2e}")
        print(f"  Tokens agreed under every seed: {', '.join(report.agreed_tokens[:10])}")
        for explanation in report.explanations:
            print(
                f"    seed {explanation.seed}: "
                f"{', '.join(entry.token for entry in explanation.top[:8])}"
            )
        print()

stability_table = pd.DataFrame(stability_rows)
stability_table

# %% [markdown]
# ### Reading the stability table
#
# The `P(phishing) spread` column must be exactly zero for every row. That
# is the evidence that the only thing varying between the two runs is LIME's
# perturbation sample, which is what makes the Jaccard column a measurement
# of LIME's own reproducibility rather than of the model's.

# %%
max_spread = stability_table["P(phishing) spread"].max()
print(f"Maximum P(phishing) spread across all cases and models: {max_spread:.2e}")
print(f"Deterministic predict path (spread == 0): {max_spread == 0.0}")
print(f"Mean top-k Jaccard overlap across seeds: {stability_table['jaccard'].mean():.4f}")
print(f"Minimum top-k Jaccard overlap: {stability_table['jaccard'].min():.4f}")

# %% [markdown]
# ## Figure: the tokens behind one verdict
#
# A horizontal bar chart of M1's top tokens for the confidently-flagged
# phishing email : the same view the desktop app renders as highlighted
# text. Bars to the right push towards phishing, bars to
# the left push towards legitimate.

# %%
figure_case = cases[0][0]
figure_explanation = explanations[figure_case]["m1"]
figure_tokens = figure_explanation.top[: config.explain.num_features]

fig, ax = plt.subplots(figsize=(8, 0.4 * len(figure_tokens) + 2))
positions = range(len(figure_tokens))
weights = [entry.weight for entry in figure_tokens]
colours = ["#b2182b" if weight > 0 else "#2166ac" for weight in weights]

ax.barh(list(positions), weights, color=colours)
ax.set_yticks(list(positions))
ax.set_yticklabels([entry.token for entry in figure_tokens])
ax.invert_yaxis()
ax.axvline(0.0, color="black", linewidth=0.8)
ax.set_xlabel("LIME weight (positive pushes towards phishing)")
ax.set_title(f"M1 Text-CNN : token attributions\n{figure_case}")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()

figures_dir = Path(config.paths.figures)
figures_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(figures_dir / "m1_lime_attributions.png", dpi=config.evaluation.figure_dpi)
plt.show()

# %% [markdown]
# ## Figure: M0 against M1 on the same email
#
# The two models' weights for the tokens either one considered important,
# side by side. Where a bar is present for one model and absent for the
# other, that token mattered to one and not the other.

# %%
m0_figure = explanations[figure_case]["m0"]
m1_figure = explanations[figure_case]["m1"]

m0_weights = {entry.token: entry.weight for entry in m0_figure.top}
m1_weights = {entry.token: entry.weight for entry in m1_figure.top}
shared_tokens = sorted(
    set(m0_weights) | set(m1_weights),
    key=lambda token: -max(abs(m0_weights.get(token, 0.0)), abs(m1_weights.get(token, 0.0))),
)[:12]

fig, ax = plt.subplots(figsize=(9, 0.45 * len(shared_tokens) + 2))
offsets = range(len(shared_tokens))
height = 0.38
ax.barh(
    [position + height / 2 for position in offsets],
    [m0_weights.get(token, 0.0) for token in shared_tokens],
    height=height,
    label="M0 TF-IDF + LogReg",
    color="#7fbf7b",
)
ax.barh(
    [position - height / 2 for position in offsets],
    [m1_weights.get(token, 0.0) for token in shared_tokens],
    height=height,
    label="M1 Text-CNN",
    color="#af8dc3",
)
ax.set_yticks(list(offsets))
ax.set_yticklabels(shared_tokens)
ax.invert_yaxis()
ax.axvline(0.0, color="black", linewidth=0.8)
ax.set_xlabel("LIME weight (positive pushes towards phishing)")
ax.set_title(f"M0 against M1 : token attributions\n{figure_case}")
ax.legend(loc="lower right")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(figures_dir / "m0_vs_m1_lime_attributions.png", dpi=config.evaluation.figure_dpi)
plt.show()

# %% [markdown]
# ## The false positive, read closely
#
# The case `notebooks/README.md` asks this notebook to produce: a legitimate
# email the model flagged, with the tokens that drove the mistake. This is
# where an explanation earns its place : the verdict alone says the model was
# wrong, the explanation says *why*, and whether the reason is a genuine
# lure-like phrase in a benign email or a corpus artefact of the kind the
# corpus probe and the LOCO evaluation exist
# to measure.

# %%
false_positive_case = next(
    (name for name in explanations if name.startswith("False positive")), None
)

if false_positive_case is None:
    print("No validation false positive at the selected threshold: nothing to review.")
else:
    bundle = explanations[false_positive_case]
    row = bundle["row"]
    explanation = bundle["m1"]
    print(f"Corpus: {row['corpus']}")
    print(f"True label: {row[label_column]} (legitimate)")
    print(f"M1 P(phishing): {explanation.probability:.4f} at threshold {selected_threshold:.4f}")
    print()
    print("Tokens that pushed this legitimate email towards phishing:")
    for entry in explanation.top:
        if entry.pushes_towards_phishing():
            print(f"  {entry.token!r:>18}  {entry.weight:+.4f}  [{entry.start}:{entry.end}]")
    print()
    print("Tokens that pushed it (correctly) towards legitimate:")
    for entry in explanation.top:
        if not entry.pushes_towards_phishing():
            print(f"  {entry.token!r:>18}  {entry.weight:+.4f}  [{entry.start}:{entry.end}]")
    print()
    print("--- The normalised text the model actually saw (first 600 characters) ---")
    print(explanation.text[:600])

# %% [markdown]
# ## Alignment check on real emails
#
# `tests/critical/test_attribution_alignment.py` proves the alignment
# invariants on fixtures. This cell asserts the same invariants on every
# real email explained above, so the notebook itself carries the evidence:
#
# 1. every offset pair slices the explained text back to its own token,
# 2. token spans are disjoint and strictly increasing,
# 3. the reported token sequence is exactly the tokeniser's own output for
#    the normalised text.

# %%
from phishlens.features.tokenise import tokenise

checked = 0
for name, bundle in explanations.items():
    for model_name in ("m0", "m1"):
        explanation = bundle[model_name]

        assert verify_offsets(explanation), f"Offset mismatch in {model_name} for {name}"

        for previous, current in zip(explanation.tokens, explanation.tokens[1:]):
            assert current.start >= previous.end, f"Overlapping spans in {model_name} for {name}"

        expected = tokenise(normalise_text(bundle["row"]["text"], config))
        assert [entry.token for entry in explanation.tokens] == expected, (
            f"Token sequence mismatch in {model_name} for {name}"
        )
        checked += 1

print(f"Alignment invariants verified on {checked} real explanations.")
print("Every offset slices back to its own token; spans are disjoint and in order.")

# %% [markdown]
# ## Summary for the report
#
# The numbers this notebook contributes to report sections 3.3 and 3.4, and
# to the XAI bonus.

# %%
summary = {
    "commit": run_commit,
    "seed": config.seed,
    "method": config.explain.method,
    "num_features": config.explain.num_features,
    "num_samples": config.explain.num_samples,
    "stability_seeds": list(config.explain.stability_seeds),
    "emails_explained": len(explanations),
    "explanations_computed": checked,
    "m1_selected_threshold": round(float(selected_threshold), 4),
    "mean_seed_jaccard": round(float(stability_table["jaccard"].mean()), 4),
    "min_seed_jaccard": round(float(stability_table["jaccard"].min()), 4),
    "max_probability_spread": float(max_spread),
    "mean_m0_m1_jaccard": round(float(agreement_table["jaccard"].mean()), 4),
}

for key, value in summary.items():
    print(f"{key}: {value}")

# %%
import json

metrics_dir = Path(config.paths.metrics)
metrics_dir.mkdir(parents=True, exist_ok=True)
explain_metrics_path = metrics_dir / "xai_stability.json"

payload = {
    **summary,
    "per_case_stability": stability_rows,
    "per_case_model_agreement": agreement_rows,
}
explain_metrics_path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)

print(f"Stability and agreement metrics written to {explain_metrics_path}")

# %% [markdown]
# ## What this notebook establishes
#
# - **Alignment is correct.** Every character offset reported for every real
#   email slices the explained text back to exactly its own token, spans are
#   disjoint and in reading order, and the token sequence is the tokeniser's
#   own output. This is asserted here on real data and proved on adversarial
#   fixtures (a repeated token, a token that is a substring of another, the
#   `<url>` and `<num>` placeholders) in
#   `tests/critical/test_attribution_alignment.py`.
# - **The explanation describes the deployed pipeline.** Both models are
#   explained through `explain/predict_fn.py`, the same wrapper the
#   `/explain` endpoint calls.
# - **Explanations are reproducible under a fixed seed, and only partly
#   reproducible across seeds.** The probability spread across seeds is
#   exactly zero, confirming the predict path is deterministic, so the
#   Jaccard overlap above measures LIME's own sampling variability and
#   nothing else. That overlap is reported as a stated limitation, not
#   smoothed over : it is the honest caveat attached to every highlighted
#   token the desktop app will show.
# - **An explanation is local.** These weights describe each individual
#   prediction, not the model's global behaviour. The global question : is
#   the model reading phishing or reading corpus artefacts : is answered by
#   the corpus probe and the leave-one-corpus-out evaluation
#, not by LIME.
