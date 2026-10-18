# Dataset card : C_PhishLens

Author: Colile Sibanda. Produced from `notebooks/01_data_audit.ipynb`;
regenerate this file's numbers from that notebook if the raw corpora ever
change.

## Source

Kaggle: [`naserabdullahalam/phishing-email-dataset`](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)

Downloaded with `scripts/download_data.py` (wraps the Kaggle API,
`kaggle.api.kaggle_api_extended.KaggleApi`) into `data/raw/`, which is
git-ignored: the corpora are public and reproducible from this link, so they
are not committed to the repository.

## Licence and access conditions

The Kaggle listing aggregates six independently sourced corpora. There is no
single licence covering the pool; each source corpus carries its own terms,
inherited from where the aggregator originally drew it:

| Corpus | Original source | Licence / access note |
|--------|-----------------|------------------------|
| CEAS-08 | CEAS 2008 Live Spam Challenge corpus | Released for the 2008 Conference on Email and Anti-Spam challenge; redistributed on Kaggle for research use |
| Enron | Enron Email Dataset (CMU) | Public-domain release resulting from the Enron litigation; long-standing research corpus |
| Ling | Ling-Spam corpus | Released for spam-filtering research (Linguist mailing list + SpamAssassin spam) |
| Nazario | Jose Nazario's phishing corpus | Public phishing-sample archive maintained for anti-phishing research |
| Nigerian Fraud | "419" advance-fee fraud sample archive | Public archive of advance-fee fraud (419 scam) emails |
| SpamAssassin | Apache SpamAssassin public corpus | Distributed under the Apache SpamAssassin project's public-corpus terms for spam-filter research |

All six are public research corpora assembled from email archives. No sample
in this dataset is ever executed, no address contained in it is ever
contacted, and no individual email is reproduced in the report without
removing personal detail.

## Corpus composition (from `notebooks/01_data_audit.ipynb`, run 2026-09-12)

Pooled row count: **82,486** across six corpora, before cleaning or splitting.

| Corpus | Legitimate (0) | Phishing/spam (1) | Total | Positive rate |
|--------|----------------|--------------------|-------|----------------|
| CEAS-08 | 17,312 | 21,842 | 39,154 | 0.558 |
| Enron | 15,791 | 13,976 | 29,767 | 0.470 |
| Ling | 2,401 | 458 | 2,859 | 0.160 |
| Nazario | 0 | 1,565 | 1,565 | 1.000 |
| Nigerian Fraud | 0 | 3,332 | 3,332 | 1.000 |
| SpamAssassin | 4,091 | 1,718 | 5,809 | 0.296 |

**Key finding:** Nazario and Nigerian Fraud contribute **only** positive
(phishing/spam) examples: neither corpus has a single legitimate row. This is
why the project measures a corpus probe and a leave-one-corpus-out evaluation
alongside the headline score: a model could learn "this email came from the
Nazario or Nigerian Fraud corpus" as a proxy for "this is phishing", which is a
corpus artefact, not a phishing signal.

## Empty and corrupted rows

- Rows with a null `body`: 1
- Rows with combined subject+body text shorter than `data.min_characters`
  (10 characters, `configs/default.yaml`): 3
- Total corrupted or empty rows (union of the two conditions): **4** out of
  82,486: negligible, dropped by `src/phishlens/data/cleaning.py`.

## Duplicate rate

- Exact duplicates on `subject` + `body` text, pooled across all six
  corpora: **0**.
- This does **not** mean the pool is leakage-free. It means no two rows are
  byte-identical on those two columns. Near-duplicate templates (for example
  Nigerian Fraud's many near-identical "urgent business assistance" emails)
  are not caught by exact matching, which is why the split groups rows by a
  normalised content hash (`src/phishlens/data/hashing.py`), not by exact text.

## Text-length distribution

Combined subject+body character length, per corpus (from the audit notebook):

| Corpus | Mean | Median | Min | Max |
|--------|------|--------|-----|-----|
| CEAS-08 | 1,608 | 598 | 21 | 144,051 |
| Enron | 1,522 | 740 | 3 | 230,135 |
| Ling | 3,254 | 2,055 | 36 | 28,672 |
| Nazario | 4,772 | 766 | 35 | 4,599,694 |
| Nigerian Fraud | 2,663 | 2,565 | 17 | 125,015 |
| SpamAssassin | 2,445 | 1,123 | 35 | 299,272 |

Nazario's maximum (4.6M characters) is a strong outlier against its own median
(766). It and one other Nazario row turned out to be mbox parser residue (a
whole mailbox landed in one cell), and cleaning drops both by their sentinel
marker rather than by length.

## Task framing

**Question:** frame the task as "phishing/spam vs legitimate" (honest to the
labels present in the pooled corpus), or restrict positives to Nazario and
Nigerian Fraud only (honest to the project's name, but confounding the positive
class with exactly two source corpora)?

**Decision:** phishing/spam vs legitimate, using the labels as provided across
all six corpora. Restricting positives to Nazario and Nigerian Fraud would tie
the entire positive class to two corpora and the entire negative class to four
others, which is precisely the corpus/label confound this project guards
against, while also discarding most of the labelled positive examples (tens of
thousands of CEAS-08 and Enron rows).

## Columns

| Column | Used as | Notes |
|--------|---------|-------|
| `subject` | Model input | Concatenated with `body` by the cleaning step |
| `body` | Model input | Primary text signal |
| `label` | Target | 1 = phishing/spam, 0 = legitimate |
| `sender`, `receiver`, `date`, `urls` | Audit and corpus-probe only | Present in CEAS-08, Nazario, Nigerian Fraud, SpamAssassin; absent in Enron and Ling. Never fed to M0 or M1, because they carry archive-specific residue |

## Known limitations

- The positive class mixes true phishing (Nazario, Nigerian Fraud) with bulk
  spam (CEAS-08, Enron, Ling, SpamAssassin), a limitation stated rather than
  hidden.
- Corpus/label confounds are present by construction (Nazario and Nigerian
  Fraud are corpus-pure positives). The corpus probe and the leave-one-corpus-out
  evaluation quantify how much this affects generalisation.
- No single licence covers the pool; each corpus's original terms apply
  (table above).
