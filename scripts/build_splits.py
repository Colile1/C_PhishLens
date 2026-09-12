# build_splits.py
# Purpose: run the data pipeline once and write the split files to data/processed/.
# Author: Colile Sibanda
#
# Orchestration only. Every transformation lives in src/phishlens/data/.

from __future__ import annotations

from pathlib import Path

from phishlens.config import load_config, set_global_seed
from phishlens.data import (
    assign_groups,
    clean,
    duplicate_report,
    load_corpora,
    make_split,
    write_split,
)


def main() -> None:
    """Build the interim frame and the train/validation/test split files.

    Inputs: none; reads configs/default.yaml and data/raw/.
    Output: none; writes data/interim/pooled_clean.csv and the split files
    plus manifest under data/processed/, and prints a summary.
    """
    config = load_config()
    set_global_seed(config.seed)

    pooled = load_corpora(config)
    print(f"Description: pooled rows loaded: {len(pooled)}")

    cleaned, report = clean(pooled, config)
    for stage, count in report.as_dict().items():
        print(f"Description: {stage}: {count}")

    grouped = assign_groups(cleaned)
    duplicates = duplicate_report(grouped)
    print(duplicates)

    interim_dir = Path(config.paths.interim)
    interim_dir.mkdir(parents=True, exist_ok=True)
    interim_path = interim_dir / "pooled_clean.csv"
    grouped.sort_values(["content_hash", "text"], kind="stable").to_csv(
        interim_path, index=False, lineterminator="\n"
    )
    print(f"Description: interim frame written: {interim_path}")

    split = make_split(grouped, config)
    write_split(split, grouped, config)

    for split_name, stats in split.manifest["realised"].items():
        print(
            f"Description: {split_name}: {stats['rows']} rows "
            f"({stats['proportion']:.4f}), {stats['groups']} groups, "
            f"positive rate {stats['positive_rate']:.4f}"
        )
    print(f"Description: config hash: {split.manifest['config_hash']}")
    print(f"Description: split files written: {config.paths.processed}")


if __name__ == "__main__":
    main()
