# splitter.py
# Purpose: grouped, stratified 70/15/15 split written once and reused everywhere.
# Author: Colile Sibanda

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from phishlens.config.schema import Config

SPLIT_NAMES = ("train", "validation", "test")
MANIFEST_FILE = "split_manifest.json"


@dataclass(frozen=True)
class SplitIndex:
    """The realised split: which content-hash group belongs to which split.

    Holds group ids rather than row indices, so the split survives any later
    re-cleaning that changes row numbering, and so the "whole group in one
    split" rule is true by construction rather than by checking.

    Inputs: one frozenset of content-hash group ids per split, plus the
    manifest describing how they were drawn.
    Output: immutable index; `split_of` maps a group id to its split name.
    """

    train: frozenset[str]
    validation: frozenset[str]
    test: frozenset[str]
    manifest: dict

    def groups_for(self, split_name: str) -> frozenset[str]:
        """Return the group ids assigned to one split.

        Inputs: split_name — one of train, validation, test.
        Output: frozenset of content-hash group ids.

        Raises ValueError naming the valid split names for anything else.
        """
        if split_name not in SPLIT_NAMES:
            raise ValueError(
                f"Unknown split '{split_name}'. Expected one of "
                f"{', '.join(SPLIT_NAMES)}."
            )
        return getattr(self, split_name)

    def split_of(self) -> dict[str, str]:
        """Invert the index into a group id to split-name lookup.

        Inputs: none.
        Output: dict mapping every assigned group id to its split name.
        """
        lookup: dict[str, str] = {}
        for split_name in SPLIT_NAMES:
            for group_id in self.groups_for(split_name):
                lookup[group_id] = split_name
        return lookup

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Attach a `split` column to a frame carrying `content_hash`.

        Inputs: frame — frame with a `content_hash` column.
        Output: a copy with a `split` column naming each row's split.

        Raises KeyError if `content_hash` is absent, and ValueError if the
        frame contains a group the index does not cover — that means the
        frame and the split were built from different data.
        """
        if "content_hash" not in frame.columns:
            raise KeyError(
                "apply() needs a 'content_hash' column. Run assign_groups() first."
            )
        lookup = self.split_of()
        assigned = frame["content_hash"].map(lookup)
        if assigned.isna().any():
            unknown = int(assigned.isna().sum())
            raise ValueError(
                f"{unknown} row(s) carry a content_hash absent from the split "
                f"index. The frame and data/processed/ were built from "
                f"different data; rebuild the split with make_split()."
            )
        result = frame.copy()
        result["split"] = assigned
        return result


def config_hash(config: Config) -> str:
    """Fingerprint the config fields that determine the split.

    Only the fields that change the split are included, so an unrelated edit
    (a learning rate, a plot DPI) does not invalidate a split that is still
    correct.

    Inputs: config — validated Config.
    Output: a 16-character hexadecimal digest.
    """
    material = json.dumps(
        {
            "seed": config.seed,
            "corpora": list(config.data.corpora),
            "text_columns": list(config.data.text_columns),
            "label_column": config.data.label_column,
            "min_characters": config.data.min_characters,
            "strip_html": config.preprocess.strip_html,
            "subject_body_separator": config.preprocess.subject_body_separator,
            "train": config.split.train,
            "validation": config.split.validation,
            "test": config.split.test,
            "stratify_by": config.split.stratify_by,
            "group_by": config.split.group_by,
        },
        sort_keys=True,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _group_table(frame: pd.DataFrame, label_column: str) -> pd.DataFrame:
    """Collapse rows into one record per content-hash group.

    Each group gets its row count and a stratum: the group's majority label,
    which is what stratification can act on, since a group is indivisible and
    may hold both labels.

    Inputs: frame — frame with `content_hash` and the label column.
    Output: DataFrame indexed by group id with columns size and stratum,
    sorted by group id so the order never depends on input row order.
    """
    grouped = frame.groupby("content_hash")[label_column]
    table = pd.DataFrame(
        {
            "size": grouped.size(),
            # Mean >= 0.5 means the group is majority positive. Ties resolve
            # to positive, deterministically.
            "stratum": (grouped.mean() >= 0.5).astype(int),
        }
    )
    return table.sort_index()


def _assign_stratum(
    group_ids: np.ndarray,
    sizes: np.ndarray,
    ratios: dict[str, float],
    rng: np.random.Generator,
) -> dict[str, list[str]]:
    """Assign one stratum's groups to splits, matching target row proportions.

    Groups are shuffled, then placed largest-first into whichever split is
    furthest below its target row count. Largest-first matters: a handful of
    very large duplicate groups dominate the row totals, and placing them
    last would leave proportions unable to recover.

    The deficit is measured *relative* to each split's target, not in absolute
    rows: train's target is more than four times validation's, so an absolute
    deficit would send almost every group to train and starve the two small
    splits. Ties are broken by a seeded draw rather than by split name, so no
    split is silently favoured by alphabetical order.

    Inputs: group_ids and sizes — parallel arrays for one stratum; ratios —
    split name to target proportion; rng — seeded generator.
    Output: dict of split name to the list of group ids assigned to it.
    """
    order = rng.permutation(len(group_ids))
    shuffled_ids = group_ids[order]
    shuffled_sizes = sizes[order]

    # Stable largest-first: ties keep the shuffled order, so the seed alone
    # decides them.
    by_size = np.argsort(-shuffled_sizes, kind="stable")

    total_rows = int(shuffled_sizes.sum())
    targets = {name: ratios[name] * total_rows for name in SPLIT_NAMES}
    placed = {name: 0.0 for name in SPLIT_NAMES}
    assignment: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}

    # One tie-breaking value per group per split, drawn up front so the
    # number of rng calls does not depend on how many ties actually occur.
    tie_breakers = rng.random((len(group_ids), len(SPLIT_NAMES)))

    for position in by_size:
        group_id = shuffled_ids[position]
        size = float(shuffled_sizes[position])
        # Fraction of its target each split still needs. A split with a zero
        # or negative ratio is never chosen.
        shortfall = {
            name: (targets[name] - placed[name]) / targets[name]
            if targets[name] > 0
            else float("-inf")
            for name in SPLIT_NAMES
        }
        chosen = max(
            SPLIT_NAMES,
            key=lambda name: (
                shortfall[name],
                float(tie_breakers[position, SPLIT_NAMES.index(name)]),
            ),
        )
        assignment[chosen].append(str(group_id))
        placed[chosen] += size

    return assignment


def make_split(frame: pd.DataFrame, config: Config) -> SplitIndex:
    """Draw the grouped, stratified train/validation/test split.

    Every content-hash group is assigned whole to exactly one split, so no
    near-duplicate email can span two splits. Stratification runs per
    majority-label stratum, so the label balance of the pool is reproduced in
    each split. The draw depends only on the config seed and the set of
    groups, never on row order.

    Inputs: frame — cleaned frame carrying `content_hash` and the label
    column; config — validated Config supplying the seed and the ratios.
    Output: SplitIndex with the three group sets and a manifest recording the
    seed, the config hash and the realised proportions.

    Raises KeyError naming any required column that is absent.
    """
    label_column = config.data.label_column
    for column in ("content_hash", label_column):
        if column not in frame.columns:
            raise KeyError(
                f"make_split() needs column '{column}', which is not in the "
                f"frame. Run clean() then assign_groups() first."
            )

    ratios = {
        "train": config.split.train,
        "validation": config.split.validation,
        "test": config.split.test,
    }
    table = _group_table(frame, label_column)
    rng = np.random.default_rng(config.seed)

    assignment: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    # Strata in sorted order so the sequence of rng draws is fixed.
    for stratum in sorted(table["stratum"].unique()):
        stratum_table = table.loc[table["stratum"] == stratum]
        stratum_assignment = _assign_stratum(
            stratum_table.index.to_numpy(),
            stratum_table["size"].to_numpy(),
            ratios,
            rng,
        )
        for name in SPLIT_NAMES:
            assignment[name].extend(stratum_assignment[name])

    group_to_split = {
        group_id: name for name in SPLIT_NAMES for group_id in assignment[name]
    }
    row_split = frame["content_hash"].map(group_to_split)
    realised = {}
    for name in SPLIT_NAMES:
        member = row_split == name
        row_count = int(member.sum())
        realised[name] = {
            "groups": len(assignment[name]),
            "rows": row_count,
            "proportion": round(row_count / len(frame), 4) if len(frame) else 0.0,
            "positive_rate": (
                round(float(frame.loc[member, label_column].mean()), 4)
                if row_count
                else 0.0
            ),
        }

    manifest = {
        "seed": config.seed,
        "config_hash": config_hash(config),
        "group_by": config.split.group_by,
        "stratify_by": config.split.stratify_by,
        "target_ratios": ratios,
        "total_rows": len(frame),
        "total_groups": len(table),
        "pooled_positive_rate": round(float(frame[label_column].mean()), 4),
        "realised": realised,
    }

    return SplitIndex(
        train=frozenset(assignment["train"]),
        validation=frozenset(assignment["validation"]),
        test=frozenset(assignment["test"]),
        manifest=manifest,
    )


def write_split(split: SplitIndex, frame: pd.DataFrame, config: Config) -> None:
    """Write the split files and the manifest to `paths.processed`.

    One CSV per split holding the modelling columns, plus the manifest. Group
    ids are sorted before writing so the files reproduce byte for byte from
    the same seed regardless of dictionary or set iteration order.

    Inputs: split — the SplitIndex; frame — the cleaned, grouped frame;
    config — validated Config supplying the output directory.
    Output: none; writes train.csv, validation.csv, test.csv and
    split_manifest.json.
    """
    processed_dir = Path(config.paths.processed)
    processed_dir.mkdir(parents=True, exist_ok=True)

    labelled = split.apply(frame)
    columns = ["text", config.data.label_column, "corpus", "content_hash"]

    for split_name in SPLIT_NAMES:
        subset = labelled.loc[labelled["split"] == split_name, columns]
        # Sort by group id, then text, so row order is a function of content
        # alone and two runs produce identical bytes.
        subset = subset.sort_values(["content_hash", "text"], kind="stable")
        subset.to_csv(
            processed_dir / f"{split_name}.csv", index=False, lineterminator="\n"
        )

    manifest_path = processed_dir / MANIFEST_FILE
    manifest_path.write_text(
        json.dumps(split.manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_split(config: Config) -> SplitIndex:
    """Read a previously written split back from `paths.processed`.

    Inputs: config — validated Config supplying the directory.
    Output: SplitIndex reconstructed from the split files and manifest.

    Raises FileNotFoundError naming the missing file and the step that
    produces it.
    """
    processed_dir = Path(config.paths.processed)
    manifest_path = processed_dir / MANIFEST_FILE

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No split manifest at {manifest_path}. Build the split first "
            "(the build, notebooks/01_data_audit.ipynb or scripts)."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    groups: dict[str, frozenset[str]] = {}
    for split_name in SPLIT_NAMES:
        split_path = processed_dir / f"{split_name}.csv"
        if not split_path.is_file():
            raise FileNotFoundError(
                f"Split file {split_path} is missing while {MANIFEST_FILE} "
                "exists. The processed directory is incomplete; rebuild the "
                "split with make_split() and write_split()."
            )
        frame = pd.read_csv(split_path, usecols=["content_hash"])
        groups[split_name] = frozenset(frame["content_hash"].astype(str))

    return SplitIndex(
        train=groups["train"],
        validation=groups["validation"],
        test=groups["test"],
        manifest=manifest,
    )
