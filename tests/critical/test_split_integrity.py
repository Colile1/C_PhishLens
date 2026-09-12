# test_split_integrity.py
# Purpose: guard that the committed split in data/processed/ leaks nothing.
# Author: Colile Sibanda

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from phishlens.config import load_config
from phishlens.data.splitter import MANIFEST_FILE, SPLIT_NAMES, config_hash, load_split

# Realised proportions and label balance are drawn per group, so they land
# near the target rather than exactly on it. One percentage point is the
# tolerance step 4 achieved; a regression past it means the draw changed.
TOLERANCE = 0.01


@pytest.fixture(scope="module")
def config():
    """Load the project config once for the module.

    Inputs: none.
    Output: validated Config from configs/default.yaml.
    """
    return load_config()


@pytest.fixture(scope="module")
def splits(config):
    """Read the three real split files from paths.processed.

    These are the files every downstream number is computed from, so the
    guards run against them rather than against a synthetic fixture.

    Inputs: config — validated Config supplying paths.processed.
    Output: dict of split name to DataFrame.

    Skips the module if the split has not been built yet, naming the script
    that builds it.
    """
    processed_dir = Path(config.paths.processed)
    frames = {}
    for split_name in SPLIT_NAMES:
        split_path = processed_dir / f"{split_name}.csv"
        if not split_path.is_file():
            pytest.skip(
                f"Split file {split_path} is missing. Build the split first: "
                "python scripts/build_splits.py"
            )
        frames[split_name] = pd.read_csv(split_path)
    return frames


@pytest.fixture(scope="module")
def pooled(splits):
    """Concatenate the three splits back into the full pool.

    Inputs: splits — dict of split name to DataFrame.
    Output: DataFrame of every row with a `split` column naming its origin.
    """
    tagged = []
    for split_name, frame in splits.items():
        copy = frame.copy()
        copy["split"] = split_name
        tagged.append(copy)
    return pd.concat(tagged, ignore_index=True)


def test_no_content_hash_group_spans_two_splits(splits):
    """No content-hash group may appear in more than one split.

    This is the central leakage guard: the pooled corpora overlap, so a group
    spanning train and test puts a near-identical email on both sides and
    inflates every reported number.
    """
    seen: dict[str, str] = {}
    offenders: list[str] = []
    for split_name in SPLIT_NAMES:
        for group_id in set(splits[split_name]["content_hash"].astype(str)):
            if group_id in seen and seen[group_id] != split_name:
                offenders.append(
                    f"group {group_id} is in both {seen[group_id]} and {split_name}"
                )
            else:
                seen[group_id] = split_name

    assert not offenders, (
        f"{len(offenders)} content-hash group(s) span more than one split, so "
        f"near-duplicate emails leak across the split boundary. First few: "
        f"{offenders[:5]}. Rebuild with python scripts/build_splits.py."
    )


def test_no_exact_duplicate_text_spans_two_splits(pooled):
    """No identical email text may appear in two different splits.

    Independent of the group check: it would still catch leakage if the
    content hash were ever changed in a way that separated identical texts.
    """
    per_text = pooled.groupby("text")["split"].nunique()
    spanning = per_text[per_text > 1]

    assert spanning.empty, (
        f"{len(spanning)} exact email text(s) appear in more than one split. "
        f"Identical text on both sides of the split boundary is direct "
        f"leakage. Rebuild with python scripts/build_splits.py."
    )


def test_row_proportions_match_targets(config, splits, pooled):
    """Each split holds its configured share of the rows, within tolerance."""
    targets = {
        "train": config.split.train,
        "validation": config.split.validation,
        "test": config.split.test,
    }
    total = len(pooled)

    for split_name in SPLIT_NAMES:
        realised = len(splits[split_name]) / total
        assert abs(realised - targets[split_name]) <= TOLERANCE, (
            f"Split '{split_name}' holds {realised:.4f} of the rows but the "
            f"config targets {targets[split_name]:.4f}, a gap wider than the "
            f"{TOLERANCE:.2f} tolerance."
        )


def test_label_proportions_preserved(config, splits, pooled):
    """Each split reproduces the pooled positive rate, within tolerance.

    A split whose class balance drifts makes its metrics incomparable with
    the others and with the pooled baseline.
    """
    label_column = config.data.label_column
    pooled_rate = float(pooled[label_column].mean())

    for split_name in SPLIT_NAMES:
        rate = float(splits[split_name][label_column].mean())
        assert abs(rate - pooled_rate) <= TOLERANCE, (
            f"Split '{split_name}' has positive rate {rate:.4f} against a "
            f"pooled {pooled_rate:.4f}, a gap wider than the {TOLERANCE:.2f} "
            f"tolerance. Stratification did not hold."
        )


def test_every_split_is_non_empty_and_covers_all_corpora(splits, pooled):
    """No split may be empty or missing a source corpus.

    A split missing a corpus would make the headline numbers and the
    leave-one-corpus-out numbers measure different populations.
    """
    all_corpora = set(pooled["corpus"].astype(str))

    for split_name in SPLIT_NAMES:
        frame = splits[split_name]
        assert len(frame) > 0, f"Split '{split_name}' is empty."
        present = set(frame["corpus"].astype(str))
        missing = all_corpora - present
        assert not missing, (
            f"Split '{split_name}' is missing corpora {sorted(missing)}, which "
            f"are present elsewhere in the pool."
        )


def test_manifest_matches_the_files(config, splits, pooled):
    """The manifest's realised counts describe the files actually on disk.

    A manifest that has drifted from the CSVs is worse than no manifest: the
    report would quote numbers no file supports.
    """
    manifest = load_split(config).manifest

    assert manifest["total_rows"] == len(pooled), (
        f"Manifest records total_rows={manifest['total_rows']} but the split "
        f"files hold {len(pooled)} rows. Rebuild with "
        f"python scripts/build_splits.py."
    )

    for split_name in SPLIT_NAMES:
        recorded = manifest["realised"][split_name]["rows"]
        actual = len(splits[split_name])
        assert recorded == actual, (
            f"Manifest records {recorded} rows for '{split_name}' but the file "
            f"holds {actual}."
        )


def test_manifest_config_hash_matches_current_config(config):
    """The split on disk was drawn from the config currently in the repo.

    If a split-determining field changed since the draw, the split is stale
    and every downstream number must be regenerated. Failing here is the only
    thing that makes that visible.
    """
    manifest = load_split(config).manifest
    expected = config_hash(config)

    assert manifest["config_hash"] == expected, (
        f"The split in data/processed/ was drawn under config_hash "
        f"{manifest['config_hash']} but the current config hashes to "
        f"{expected}. A split-determining field changed; rebuild with "
        f"python scripts/build_splits.py and regenerate every downstream "
        f"number."
    )


def test_manifest_records_grouped_and_stratified_draw(config):
    """The manifest states the split was grouped by content hash on the seed.

    Guards against a future session quietly replacing the grouped draw with a
    plain random one, which would pass every disjointness check above while
    leaking near-duplicates.
    """
    manifest = load_split(config).manifest
    processed_dir = Path(config.paths.processed)

    assert (processed_dir / MANIFEST_FILE).is_file()
    assert manifest["group_by"] == config.split.group_by == "content_hash", (
        f"Manifest records group_by={manifest['group_by']!r}; the split must "
        f"be grouped by content_hash or near-duplicates leak across splits."
    )
    assert manifest["stratify_by"] == config.split.stratify_by
    assert manifest["seed"] == config.seed, (
        f"Manifest seed {manifest['seed']} does not match the config seed "
        f"{config.seed}."
    )
