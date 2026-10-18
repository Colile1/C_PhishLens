# docs/screenshots/

The project in pictures, in the order the work happened: the data, the
baseline, training, the one-time test evaluation, the generalisation check, the
explanations, and then the three ways to use the model (browser, desktop,
notebook), the API, the tests and the repository. Notebook shots are taken from
the committed notebooks; the browser, API and GitHub shots from the live
deployment; the desktop shots from the running Tkinter app.

## 1. The data

| Screenshot | What it shows |
|---|---|
| [01-data-class-balance.png](01-data-class-balance.png) | Rows and class balance per source corpus. Nazario and Nigerian Fraud contain only phishing, which is why corpus recognition is a risk |
| [02-data-duplicate-rate.png](02-data-duplicate-rate.png) | Zero exact duplicates, and why near-duplicates still need content-hash grouping |
| [03-data-text-length.png](03-data-text-length.png) | Text length per corpus, including the parser-residue outliers that cleaning removes |

## 2. Baseline and training

| Screenshot | What it shows |
|---|---|
| [04-baseline-metrics.png](04-baseline-metrics.png) | M0 (TF-IDF + logistic regression) validation metrics: the number M1 has to beat |
| [05-baseline-figures.png](05-baseline-figures.png) | M0 ROC, precision-recall and confusion matrix |
| [06-training-loop-with-patience.png](06-training-loop-with-patience.png) | The training loop printed in the notebook: patience on validation macro-F1 and best-weight restore |
| [07-training-loss-curve.png](07-training-loss-curve.png) | Training and validation loss per epoch; validation loss turns up after epoch 3 |
| [08-training-per-epoch-log.png](08-training-per-epoch-log.png) | Every metric for every epoch (accuracy, precision, recall, F1, macro-F1, AUC-ROC, loss) |

## 3. Test-set evaluation (run once)

| Screenshot | What it shows |
|---|---|
| [09-test-m0-vs-m1.png](09-test-m0-vs-m1.png) | M0 against M1 on the test split at each model's validation-selected threshold |
| [10-test-figures-m1.png](10-test-figures-m1.png) | M1 ROC curve, precision-recall curve, confusion matrix and loss curve |
| [11-test-per-corpus.png](11-test-per-corpus.png) | M1 test results broken down by source corpus |

## 4. Does it generalise?

| Screenshot | What it shows |
|---|---|
| [12-loco-results.png](12-loco-results.png) | Leave-one-corpus-out: M1 trained on five corpora, tested on the sixth, for each corpus |
| [13-loco-vs-headline-vs-probe.png](13-loco-vs-headline-vs-probe.png) | The headline score against the leave-one-corpus-out mean and the corpus-probe accuracy, in one place |

## 5. Explanations

| Screenshot | What it shows |
|---|---|
| [14-xai-lime-tokens.png](14-xai-lime-tokens.png) | LIME: the words behind M1's verdict on a confidently flagged phishing email |
| [15-xai-false-positive.png](15-xai-false-positive.png) | A legitimate email the model flagged, and the words that caused the mistake |
| [16-xai-lime-vs-integrated-gradients.png](16-xai-lime-vs-integrated-gradients.png) | LIME cross-checked with Integrated Gradients: agreement per email and the side-by-side figure |

## 6. Using the model

**In a notebook** (`notebooks/07_analyst_ui.ipynb`, no server needed)

| Screenshot | What it shows |
|---|---|
| [17-notebook-ui-phishing.png](17-notebook-ui-phishing.png) | `analyse(text)` on a credential lure: verdict, highlighted words, strongest signals |
| [18-notebook-ui-own-email.png](18-notebook-ui-own-email.png) | The "analyse your own email" cell on a parcel-delivery lure |

**In the browser** (live at https://phishlens-vdc9.onrender.com)

| Screenshot | What it shows |
|---|---|
| [19-live-ui-ready.png](19-live-ui-ready.png) | The Analyse page with the API status badge |
| [20-live-ui-phishing-verdict.png](20-live-ui-phishing-verdict.png) | A phishing verdict: probability against the threshold, highlighted words, signals table |
| [21-live-ui-legitimate-verdict.png](21-live-ui-legitimate-verdict.png) | A legitimate verdict on an ordinary meeting email |
| [22-live-ui-error-message.png](22-live-ui-error-message.png) | An over-long email refused with the API's own reason |
| [23-live-model-provenance.png](23-live-model-provenance.png) | The Model page: which model is served, its commit and vocabulary hash, and its known limitations |
| [24-live-user-guide.png](24-live-user-guide.png) | The built-in user guide and glossary |
| [25-live-ui-phone-width.png](25-live-ui-phone-width.png) | The same flow at phone width |

**On the desktop** (Tkinter, `python run_app.py`)

| Screenshot | What it shows |
|---|---|
| [../report/desktop-analysed-phishing.png](../report/desktop-analysed-phishing.png) | A phishing email analysed: verdict banner, probability meter, highlights, signals |
| [31-desktop-legitimate-verdict.png](31-desktop-legitimate-verdict.png) | A legitimate meeting email analysed |
| [32-desktop-parcel-lure.png](32-desktop-parcel-lure.png) | A parcel-delivery lure flagged at 99.4% |
| [../report/desktop-api-down.png](../report/desktop-api-down.png) | The API stopped: a readable message instead of a crash |

## 7. The API, the tests and the repository

| Screenshot | What it shows |
|---|---|
| [26-api-interactive-docs.png](26-api-interactive-docs.png) | The four endpoints in the live interactive API docs |
| [27-api-metadata-provenance.png](27-api-metadata-provenance.png) | `/metadata`: model id, threshold, seed, training commit, vocabulary hash, dataset version |
| [30-tests-and-notebook-audit.png](30-tests-and-notebook-audit.png) | The 84 guard tests and the full suite of 210 passing, and the notebook audit passing |
| [28-github-repository.png](28-github-repository.png) | The repository on GitHub |
| [29-github-commit-history.png](29-github-commit-history.png) | The commit history: small commits across the project's eleven working days |
