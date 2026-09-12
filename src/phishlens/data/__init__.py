# __init__.py
# Purpose: public entry point for the data package.
# Author: Colile Sibanda

from phishlens.data.cleaning import CleaningReport, clean, strip_html
from phishlens.data.dedup import assign_groups, duplicate_report
from phishlens.data.hashing import content_hash, normalise_for_hash
from phishlens.data.loader import load_corpora
from phishlens.data.loco import LocoFold, loco_report, make_loco_index
from phishlens.data.splitter import (
    SPLIT_NAMES,
    SplitIndex,
    config_hash,
    load_split,
    make_split,
    write_split,
)

__all__ = [
    "load_corpora",
    "clean",
    "strip_html",
    "CleaningReport",
    "content_hash",
    "normalise_for_hash",
    "assign_groups",
    "duplicate_report",
    "make_split",
    "write_split",
    "load_split",
    "config_hash",
    "SplitIndex",
    "SPLIT_NAMES",
    "make_loco_index",
    "loco_report",
    "LocoFold",
]
