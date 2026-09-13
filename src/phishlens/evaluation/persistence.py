# persistence.py
# Purpose: write a metric JSON that traces back to the run that produced it.
# Author: Colile Sibanda

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from phishlens.evaluation.metrics import Metrics

# Metric values are rounded before writing so the JSON stays readable and the
# report quotes a stable figure. Four places is what artifacts/metrics/
# m0_baseline.json already uses; changing it would make the M0 and M1 files
# disagree on precision for no gain.
METRIC_PRECISION = 4

UNKNOWN_COMMIT = "unknown"


def commit_hash() -> str:
    """Return the current git commit, or "unknown" outside a repository.

    Every metric file records the commit that produced it, so a number in the
    report can be traced to the code that computed it. This returns a sentinel
    rather than raising when git is unavailable — a reader running from an
    extracted zip has no repository, and failing there would block the
    notebooks from running at all.

    Inputs: none.
    Output: the full commit SHA, or "unknown" if git is absent or the working
    directory is not a repository.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return UNKNOWN_COMMIT

    return completed.stdout.strip() or UNKNOWN_COMMIT


def save_metrics(metrics: Metrics, path: Path, context: dict[str, object]) -> None:
    """Write a metric JSON with its provenance embedded.

    The scores alone are not enough to defend a number in the report. The
    context carries what makes it reproducible: which model, which split, the
    seed, the commit and the config hash. The config hash is passed in rather
    than computed here because it lives in `data.splitter` and this package

    The threshold is written as a top-level key from the Metrics itself, not
    from the context, so it cannot be reported as something other than the
    threshold the scores were actually computed at.

    Inputs: metrics — the scores to record; path — destination JSON file;
    context — provenance fields written flat alongside the scores, typically
    model, split_evaluated, seed, commit_hash, config_hash and row counts.
    Output: none; writes the file, creating parent directories as needed.

    Raises ValueError if the context supplies its own "metrics" or "threshold"
    key, which would silently contradict the Metrics being written.
    """
    reserved = {"metrics", "threshold"} & set(context)
    if reserved:
        raise ValueError(
            f"save_metrics() got reserved context key(s) {sorted(reserved)!r}. "
            f"Both are written from the Metrics itself so they cannot disagree "
            f"with the scores; remove them from the context."
        )

    payload: dict[str, object] = dict(context)
    payload["threshold"] = metrics.threshold
    payload["metrics"] = {
        name: round(float(value), METRIC_PRECISION)
        for name, value in metrics.as_dict().items()
    }
    payload["confusion_matrix"] = {
        "true_negatives": metrics.true_negatives,
        "false_positives": metrics.false_positives,
        "false_negatives": metrics.false_negatives,
        "true_positives": metrics.true_positives,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
