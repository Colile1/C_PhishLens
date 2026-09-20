# test_data.py
# Purpose: unit tests for the cleaning, hashing, dedup and splitting layer.
# Author: Colile Sibanda

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from phishlens.config import load_config
from phishlens.data import (
    assign_groups,
    clean,
    content_hash,
    duplicate_report,
    load_corpora,
    make_loco_index,
    make_split,
    normalise_for_hash,
    strip_html,
    write_split,
)
from phishlens.data.cleaning import MBOX_SENTINEL
from phishlens.data.splitter import SPLIT_NAMES, load_split


@pytest.fixture(scope="module")
def config():
    """Load the project config once for the module.

    Inputs: none.
    Output: validated Config from configs/default.yaml.
    """
    return load_config()


def _synthetic_frame() -> pd.DataFrame:
    """Build a small frame with known duplicates, labels and corpora.

    Inputs: none.
    Output: DataFrame with subject, body, label, corpus columns.
    """
    # Family names are words, not digits: the hash strips digits by design,
    # so distinct families must differ in their wording to stay distinct.
    topics = [
        "account", "invoice", "delivery", "password", "refund", "payroll",
        "security", "subscription", "warranty", "storage", "meeting", "receipt",
    ]
    rows = []
    # Twelve template families, each repeated a different number of times, so
    # group sizes vary the way they do in the real pool.
    for family, topic in enumerate(topics):
        repeats = 1 + (family % 4)
        for copy in range(repeats):
            rows.append(
                {
                    "subject": f"Your {topic} notice",
                    # Only the digits change between copies of a family :
                    # exactly what the hash must ignore.
                    "body": f"Regarding your {topic} record {copy * 17}, "
                    f"reference {copy} is ready. Please confirm now.",
                    "label": family % 2,
                    "corpus": f"Corpus_{family % 3}",
                }
            )
    return pd.DataFrame(rows)


class TestHashing:
    """Content hashing must ignore per-victim substitutions but not wording."""

    def test_near_duplicates_share_a_hash(self):
        """Emails differing only in amount, date and reference hash the same."""
        first = "Dear customer, your invoice of $1,250 dated 03/04/2021 is due."
        second = "Dear customer, your invoice of $9,870 dated 15/11/2023 is due."
        assert content_hash(first) == content_hash(second)

    def test_a_substituted_name_is_not_merged(self):
        """A per-victim name change does NOT collapse two emails into one group.

        Deliberate and conservative: merging on alphabetic differences would
        need a name list and would over-merge genuinely distinct emails.
        Splitting a template family across splits is the milder failure;
        merging unrelated emails would shrink the data and hide variation.
        """
        first = "Dear Alice, your invoice of $1,250 is due."
        second = "Dear Bob, your invoice of $9,870 is due."
        assert content_hash(first) != content_hash(second)

    def test_different_wording_differs(self):
        """A genuinely different email gets a different hash."""
        first = "Your package could not be delivered, confirm your address."
        second = "Quarterly revenue figures are attached for your review."
        assert content_hash(first) != content_hash(second)

    def test_case_and_whitespace_insensitive(self):
        """Casing and whitespace do not change the group."""
        assert content_hash("Verify   Your\nAccount") == content_hash(
            "verify your account"
        )

    def test_normalise_is_idempotent(self):
        """Normalising an already-normalised string changes nothing."""
        once = normalise_for_hash("Pay $400 NOW!! <b>urgent</b>")
        assert normalise_for_hash(once) == once

    def test_non_string_input_is_safe(self):
        """A null body does not raise; it hashes as empty content."""
        assert content_hash(None) == content_hash("")


class TestCleaning:
    """Cleaning must build the text column and drop only unusable rows."""

    def test_strip_html_removes_tags_and_script(self):
        """Tags go, script bodies go, visible text and entities survive."""
        cleaned = strip_html(
            "<p>Click <a href='x'>here</a> &amp; win</p><script>evil()</script>"
        )
        assert "evil" not in cleaned
        assert "<" not in cleaned
        assert "Click" in cleaned and "here" in cleaned and "&" in cleaned

    def test_mbox_sentinel_rows_are_dropped(self, config):
        """Parser residue carrying the mbox sentinel is removed."""
        frame = pd.DataFrame(
            {
                "subject": ["Real email", "Residue"],
                "body": [
                    "This is a perfectly ordinary message body here.",
                    f"{MBOX_SENTINEL} this text is part of the internal format.",
                ],
                "label": [0, 1],
                "corpus": ["A", "A"],
            }
        )
        cleaned, report = clean(frame, config)
        assert report.dropped_mbox_sentinel == 1
        assert len(cleaned) == 1
        assert cleaned.loc[0, "subject"] == "Real email"

    def test_null_and_non_binary_labels_are_dropped(self, config):
        """Only labels 0 and 1 survive; the report counts the rest."""
        frame = pd.DataFrame(
            {
                "subject": ["a", "b", "c", "d"],
                "body": ["long enough body one", "long enough body two",
                         "long enough body three", "long enough body four"],
                "label": [0, 1, None, 7],
                "corpus": ["A"] * 4,
            }
        )
        cleaned, report = clean(frame, config)
        assert report.dropped_null_label == 2
        assert set(cleaned["label"]) == {0, 1}
        assert cleaned["label"].dtype.kind == "i"

    def test_short_rows_are_dropped(self, config):
        """Rows below min_characters are dropped and counted."""
        frame = pd.DataFrame(
            {
                "subject": ["", "Proper subject"],
                "body": ["", "A body comfortably longer than the minimum."],
                "label": [1, 0],
                "corpus": ["A", "A"],
            }
        )
        cleaned, report = clean(frame, config)
        assert report.dropped_too_short == 1
        assert len(cleaned) == 1

    def test_missing_column_names_itself(self, config):
        """A frame without the label column raises an error naming it."""
        frame = pd.DataFrame({"subject": ["a"], "body": ["b"]})
        with pytest.raises(KeyError, match="label"):
            clean(frame, config)

    def test_report_counts_reconcile(self, config):
        """rows_in minus every drop equals rows_out."""
        frame = _synthetic_frame()
        _, report = clean(frame, config)
        assert (
            report.rows_in
            - report.dropped_null_label
            - report.dropped_mbox_sentinel
            - report.dropped_too_short
            == report.rows_out
        )


class TestDedup:
    """Grouping attaches ids without deleting rows."""

    def test_assign_groups_keeps_every_row(self, config):
        """No row is lost, and a content_hash is attached to each."""
        cleaned, _ = clean(_synthetic_frame(), config)
        grouped = assign_groups(cleaned)
        assert len(grouped) == len(cleaned)
        assert grouped["content_hash"].notna().all()

    def test_template_families_collapse(self, config):
        """The twelve template families collapse to twelve groups."""
        cleaned, _ = clean(_synthetic_frame(), config)
        grouped = assign_groups(cleaned)
        assert grouped["content_hash"].nunique() == 12

    def test_duplicate_report_totals(self, config):
        """The pooled ALL row matches the frame it describes."""
        cleaned, _ = clean(_synthetic_frame(), config)
        grouped = assign_groups(cleaned)
        report = duplicate_report(grouped)
        assert report.loc["ALL", "rows"] == len(grouped)
        assert report.loc["ALL", "unique_groups"] == 12

    def test_missing_text_column_names_itself(self):
        """Calling assign_groups before clean raises a named error."""
        with pytest.raises(KeyError, match="text"):
            assign_groups(pd.DataFrame({"body": ["x"]}))


class TestSplitter:
    """The split must be grouped, stratified, proportioned and reproducible."""

    @pytest.fixture()
    def grouped(self, config):
        """Cleaned and grouped synthetic frame.

        Inputs: config fixture.
        Output: frame with text, label, corpus and content_hash.
        """
        cleaned, _ = clean(_synthetic_frame(), config)
        return assign_groups(cleaned)

    def test_no_group_spans_two_splits(self, grouped, config):
        """Every content-hash group lands wholly in one split."""
        split = make_split(grouped, config)
        assert not (split.train & split.validation)
        assert not (split.train & split.test)
        assert not (split.validation & split.test)

    def test_every_group_is_assigned(self, grouped, config):
        """The three splits together cover every group exactly once."""
        split = make_split(grouped, config)
        covered = split.train | split.validation | split.test
        assert covered == set(grouped["content_hash"])
        assert len(split.train) + len(split.validation) + len(split.test) == len(
            covered
        )

    def test_same_seed_reproduces_the_split(self, grouped, config):
        """Two draws with the same seed produce identical group sets."""
        first = make_split(grouped, config)
        second = make_split(grouped, config)
        assert first.train == second.train
        assert first.validation == second.validation
        assert first.test == second.test

    def test_row_order_does_not_change_the_split(self, grouped, config):
        """Shuffling the input rows produces the same split."""
        first = make_split(grouped, config)
        shuffled = grouped.sample(frac=1.0, random_state=7).reset_index(drop=True)
        second = make_split(shuffled, config)
        assert first.train == second.train
        assert first.test == second.test

    def test_a_different_seed_changes_the_split(self, config):
        """The seed genuinely drives the draw.

        Uses many single-row groups, because with only a handful of large
        groups the proportion-filling leaves no freedom for the seed to
        express and both draws legitimately coincide.
        """
        # Distinct wording per row, since digits are stripped before hashing.
        alphabet = "abcdefghijklmnopqrst"
        words = [
            f"{first}{second}" for first in alphabet for second in alphabet
        ][:200]
        frame = pd.DataFrame(
            {
                "subject": [f"Notice concerning {word}" for word in words],
                "body": [
                    f"This message about {word} is long enough to survive "
                    f"the cleaning step without being dropped."
                    for word in words
                ],
                "label": [n % 2 for n in range(200)],
                "corpus": ["A"] * 200,
            }
        )
        cleaned, _ = clean(frame, config)
        grouped = assign_groups(cleaned)
        other = dataclasses.replace(config, seed=config.seed + 1)
        assert make_split(grouped, config).test != make_split(grouped, other).test

    def test_realised_proportions_are_within_one_point(self, config):
        """Row proportions land within 1 percentage point of 70/15/15.

        Guards the failure where the deficit is measured in absolute rows:
        train's target dwarfs the other two, so every group goes to train and
        validation and test come back empty.
        """
        alphabet = "abcdefghijklmnopqrst"
        words = [f"{a}{b}" for a in alphabet for b in alphabet][:300]
        frame = pd.DataFrame(
            {
                "subject": [f"Notice concerning {word}" for word in words],
                "body": [
                    f"This message about {word} is long enough to survive "
                    f"the cleaning step without being dropped."
                    for word in words
                ],
                "label": [n % 2 for n in range(300)],
                "corpus": ["A"] * 300,
            }
        )
        cleaned, _ = clean(frame, config)
        grouped = assign_groups(cleaned)
        split = make_split(grouped, config)

        targets = {
            "train": config.split.train,
            "validation": config.split.validation,
            "test": config.split.test,
        }
        for split_name, target in targets.items():
            realised = split.manifest["realised"][split_name]["proportion"]
            assert abs(realised - target) <= 0.01, (
                f"{split_name} proportion {realised} is more than one "
                f"percentage point from its target {target}"
            )

    def test_label_balance_is_preserved(self, config):
        """Each split's positive rate tracks the pool's within 1 point."""
        alphabet = "abcdefghijklmnopqrst"
        words = [f"{a}{b}" for a in alphabet for b in alphabet][:300]
        frame = pd.DataFrame(
            {
                "subject": [f"Notice concerning {word}" for word in words],
                "body": [
                    f"This message about {word} is long enough to survive "
                    f"the cleaning step without being dropped."
                    for word in words
                ],
                "label": [n % 2 for n in range(300)],
                "corpus": ["A"] * 300,
            }
        )
        cleaned, _ = clean(frame, config)
        grouped = assign_groups(cleaned)
        split = make_split(grouped, config)

        pooled_rate = split.manifest["pooled_positive_rate"]
        for split_name in SPLIT_NAMES:
            rate = split.manifest["realised"][split_name]["positive_rate"]
            assert abs(rate - pooled_rate) <= 0.01, (
                f"{split_name} positive rate {rate} differs from the pooled "
                f"rate {pooled_rate} by more than one percentage point"
            )

    def test_apply_rejects_an_unknown_group(self, grouped, config):
        """A frame built from different data is refused, not silently split."""
        split = make_split(grouped, config)
        stranger = grouped.head(1).copy()
        stranger["content_hash"] = "0" * 64
        with pytest.raises(ValueError, match="content_hash"):
            split.apply(pd.concat([grouped, stranger], ignore_index=True))

    def test_write_and_load_round_trip(self, grouped, config, tmp_path):
        """A written split reads back as the same group sets."""
        local = dataclasses.replace(
            config,
            paths=dataclasses.replace(config.paths, processed=str(tmp_path)),
        )
        split = make_split(grouped, local)
        write_split(split, grouped, local)
        reloaded = load_split(local)
        assert reloaded.train == split.train
        assert reloaded.validation == split.validation
        assert reloaded.test == split.test
        assert reloaded.manifest["config_hash"] == split.manifest["config_hash"]

    def test_written_files_are_byte_identical_across_runs(
        self, grouped, config, tmp_path
    ):
        """Re-running with the same seed reproduces the files byte for byte."""
        digests = []
        for run in ("a", "b"):
            target = tmp_path / run
            local = dataclasses.replace(
                config,
                paths=dataclasses.replace(config.paths, processed=str(target)),
            )
            write_split(make_split(grouped, local), grouped, local)
            digests.append(
                {
                    path.name: path.read_bytes()
                    for path in sorted(target.iterdir())
                }
            )
        assert digests[0] == digests[1]

    def test_missing_split_directory_names_the_step(self, config, tmp_path):
        """Loading before building raises an error that says what to run."""
        local = dataclasses.replace(
            config,
            paths=dataclasses.replace(config.paths, processed=str(tmp_path)),
        )
        with pytest.raises(FileNotFoundError, match="manifest"):
            load_split(local)


class TestLoco:
    """Leave-one-corpus-out folds must hold out a corpus genuinely."""

    @pytest.fixture()
    def grouped(self, config):
        """Cleaned and grouped synthetic frame.

        Inputs: config fixture.
        Output: frame with content_hash and corpus.
        """
        cleaned, _ = clean(_synthetic_frame(), config)
        return assign_groups(cleaned)

    def test_one_fold_per_corpus(self, grouped):
        """There is exactly one fold per source corpus."""
        folds = make_loco_index(grouped)
        assert set(folds) == set(grouped["corpus"].unique())

    def test_train_and_test_never_share_a_group(self, grouped):
        """Shared content is removed from the train side of every fold."""
        for fold in make_loco_index(grouped).values():
            assert not (fold.train & fold.test)
