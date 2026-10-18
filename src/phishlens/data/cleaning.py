# cleaning.py
# Purpose: turn pooled raw rows into modellable text and report what was dropped.
# Author: Colile Sibanda

from __future__ import annotations

import html
import re
from dataclasses import dataclass

import pandas as pd

from phishlens.config.schema import Config

# Two rows in Nazario are not emails: they are mbox parser residue in which an
# entire mailbox, headers and all, landed in one cell (notebooks/01_data_audit.ipynb found them
# as 4.6M- and 1.0M-character outliers against a 766-character median). Both
# carry this sentinel, which the mail system writes at the head of a folder's
# internal-data record. No other corpus contains it. Dropping on the sentinel is
# exact; dropping on length would discard genuinely long emails elsewhere in
# the pool, where bodies reach 300k characters legitimately.
MBOX_SENTINEL = "FOLDER INTERNAL DATA"

_SCRIPT_STYLE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class CleaningReport:
    """Counts of what cleaning removed, for the audit notebook and the manifest.

    Inputs: the row counts observed at each drop stage.
    Output: immutable record; `as_dict` renders it for JSON.
    """

    rows_in: int
    dropped_null_label: int
    dropped_mbox_sentinel: int
    dropped_too_short: int
    rows_out: int

    def as_dict(self) -> dict[str, int]:
        """Render the report as a plain dictionary for JSON serialisation.

        Inputs: none.
        Output: dict of stage name to row count.
        """
        return {
            "rows_in": self.rows_in,
            "dropped_null_label": self.dropped_null_label,
            "dropped_mbox_sentinel": self.dropped_mbox_sentinel,
            "dropped_too_short": self.dropped_too_short,
            "rows_out": self.rows_out,
        }


def strip_html(text: str) -> str:
    """Remove HTML markup and unescape entities, keeping the visible text.

    Script and style elements are removed with their contents, because their
    bodies are code rather than anything the recipient read.

    Inputs: text : raw email text, possibly containing HTML.
    Output: the visible text with whitespace collapsed.
    """
    if not isinstance(text, str):
        return ""
    without_code = _SCRIPT_STYLE.sub(" ", text)
    without_tags = _HTML_TAG.sub(" ", without_code)
    return html.unescape(without_tags)


def clean(frame: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, CleaningReport]:
    """Build the model's `text` column and drop rows that are not usable emails.

    Joins the configured text columns with the configured separator, strips
    HTML when enabled, collapses whitespace, and drops in order: rows with a
    null or non-binary label, mbox parser residue, and rows shorter than
    `data.min_characters`. Lowercasing is deliberately NOT applied here : that
    belongs to the feature layer, so this frame stays readable in the audit and
    in explanations.

    Inputs: frame : pooled frame from `load_corpora`; config : validated Config.
    Output: (cleaned frame with a `text` column and a reset index,
    CleaningReport of what was dropped).

    Raises KeyError naming any configured text or label column not present.
    """
    text_columns = list(config.data.text_columns)
    label_column = config.data.label_column

    absent = [
        column
        for column in text_columns + [label_column]
        if column not in frame.columns
    ]
    if absent:
        raise KeyError(
            f"clean() requires column(s) not present in the frame: "
            f"{', '.join(absent)}. Check data.text_columns and "
            f"data.label_column in configs/default.yaml."
        )

    working = frame.copy()
    rows_in = len(working)

    # Labels must be a clean binary 0/1. Anything else is a corrupted row, not
    # a class to guess at.
    labels = pd.to_numeric(working[label_column], errors="coerce")
    valid_label = labels.isin([0, 1])
    dropped_null_label = int((~valid_label).sum())
    working = working.loc[valid_label].copy()
    working[label_column] = labels.loc[valid_label].astype(int)

    joined = (
        working[text_columns]
        .fillna("")
        .astype(str)
        .agg(config.preprocess.subject_body_separator.join, axis=1)
    )

    is_residue = joined.str.contains(MBOX_SENTINEL, case=False, regex=False, na=False)
    dropped_mbox_sentinel = int(is_residue.sum())
    working = working.loc[~is_residue].copy()
    joined = joined.loc[~is_residue]

    if config.preprocess.strip_html:
        joined = joined.map(strip_html)
    joined = joined.str.replace(_WHITESPACE, " ", regex=True).str.strip()

    working["text"] = joined

    long_enough = working["text"].str.len() >= config.data.min_characters
    dropped_too_short = int((~long_enough).sum())
    working = working.loc[long_enough].reset_index(drop=True)

    report = CleaningReport(
        rows_in=rows_in,
        dropped_null_label=dropped_null_label,
        dropped_mbox_sentinel=dropped_mbox_sentinel,
        dropped_too_short=dropped_too_short,
        rows_out=len(working),
    )
    return working, report
