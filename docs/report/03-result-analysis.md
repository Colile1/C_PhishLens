# 3. Model result analysis

Every figure referenced here is rendered inline in the notebooks named. Every
number is read from the committed artefacts in `artifacts/metrics/`, not from a
later or unrecorded run.

## 3.1 Headline results on the random split

M1 was trained on 57 736 emails, with 12 372 held out for validation and 12 373
for test. Splitting is grouped on content hash, so near-identical emails cannot
appear in two splits at once. The decision threshold was selected on validation
and applied unchanged to test.

| Metric | M0 TF-IDF + LogReg | M1 Text-CNN |
|--------|--------------------|-------------|
| Accuracy | 0.9856 | **0.9871** |
| Precision | 0.9857 | 0.9829 |
| Recall | 0.9866 | **0.9925** |
| F1 | 0.9862 | **0.9877** |
| Macro-F1 | 0.9856 | **0.9871** |
| AUC-ROC | 0.9988 | **0.9991** |
| Average precision | 0.9989 | **0.9992** |
| Threshold | 0.5055 | 0.5698 |

Confusion matrices on the 12 373 test emails:

| | M0 | M1 |
|---|---|---|
| True positives | 6 347 | 6 385 |
| True negatives | 5 848 | 5 829 |
| False positives | 92 | 111 |
| False negatives | 86 | **48** |

Figures 1 to 4 (`notebooks/04_evaluate.ipynb`) show the M1 ROC curve,
precision-recall curve, confusion matrix and loss curve. The M0 equivalents are
saved in `artifacts/figures/` and shown in `notebooks/02_baseline.ipynb`.

Two observations. First, the deep model beats the linear baseline, but by
0.0015 macro-F1. On a 12 373-row test set that is a difference of roughly
nineteen emails, which is not a decisive margin. A TF-IDF bag of bigrams is
close to sufficient for this task as posed, and that fact is itself a result:
it suggests the discriminating signal is largely lexical presence rather than
composition, which is what section 3.4 then explains.

Second, the models differ in the shape of their errors, not their quantity. M1
trades 19 additional false positives for 38 fewer false negatives, raising
recall from 0.9866 to 0.9925 at a small cost in precision. For the triage
scenario in section 1, where a missed phishing message is more costly than a
legitimate message read one place too early in the queue, that is the preferable
trade, and it is the reason M1 is the deliverable model even though its headline
advantage is slight.

## 3.2 Training behaviour and early stopping

The per-epoch log is `artifacts/metrics/training_log.csv`, printed as a table in
`notebooks/03_train_cnn.ipynb` and plotted as Figure 4.

| Epoch | Train loss | Val loss | Val macro-F1 |
|-------|-----------|----------|--------------|
| 1 | 0.1948 | 0.0783 | 0.9707 |
| 2 | 0.0765 | 0.0441 | 0.9829 |
| 3 | 0.0416 | 0.0405 | 0.9859 |
| 4 | 0.0270 | 0.0409 | 0.9862 |
| 5 | 0.0185 | 0.0524 | 0.9845 |
| 6 | 0.0137 | 0.0510 | **0.9867** |

The curves show textbook overfitting onset. Training loss falls monotonically to
0.0137, while validation loss reaches its minimum of 0.0405 at epoch 3 and rises
thereafter. From epoch 4 onward the model is fitting the training set rather
than learning transferable signal.

This is exactly the case where stopping on validation loss and stopping on
validation macro-F1 disagree, and it justifies the design choice made in section
2.4. Validation loss says epoch 3 and is worsening by epoch 6; validation
macro-F1 nonetheless peaks at epoch 6 (0.98665, narrowly above epoch 4's
0.98624). Because patience watches macro-F1 with `min_delta = 0.001`, epoch 6's
improvement over epoch 4 of 0.0004 did **not** count as an improvement, the
patience counter continued to run from epoch 4, and training halted. The
restored weights are the best-macro-F1 checkpoint. The reported test numbers
above are from that restored checkpoint, not from the final epoch.

The wider lesson is that a well-calibrated loss and a good decision metric are
not the same objective, and reporting whichever one flatters the model is the
error that patience-on-the-right-metric exists to prevent.

## 3.3 The corpus probe: what the headline number partly measures

The headline figures above are flattering, and the rest of this section explains
why. The concern is drawn from Bhuiyan & Bhuiyan (2026), who showed that on a
pooled phishing corpus a linear model can identify which *source archive* an
email came from with 97-98% accuracy. A phishing classifier trained on such a
pool can therefore score well by recognising corpus markers rather than
phishing.

This project measured the effect directly rather than citing it. A TF-IDF and
logistic regression probe was trained to predict an email's source corpus from
its text alone, on the same splits, using the same text-only constraint as the
classifier (`tests/critical/`, `artifacts/metrics/corpus_probe.json`).

**The probe reaches 0.9445 accuracy and 0.9081 macro-F1 across six corpora.**

| Held-out corpus | Probe F1 |
|-----------------|----------|
| CEAS_08 | 0.9731 |
| Enron | 0.9423 |
| Nazario | 0.9213 |
| Nigerian_Fraud | 0.8965 |
| Ling | 0.8743 |
| SpamAssasin | 0.8410 |

A six-way classification at 0.9445 accuracy, against a majority-class baseline
far below that, means the text of these emails identifies its archive almost as
reliably as it identifies its label. The corpora are separable by collection era
conventions, mailing-list boilerplate, date and formatting residue, and
vocabulary drift between 1990s and 2000s mail. The implication is unavoidable
and is stated here as a limitation rather than buried: **part of the 0.9871
headline macro-F1 is corpus recognition, not phishing detection.** Because
phishing and legitimate mail in this pool come from largely different source
archives, a model that learns to recognise the archive gets the label right for
free.

Reproducing the effect at 0.9445 rather than the paper's 0.97-0.98 is consistent
with that work, and the gap is plausibly explained by the text-only constraint
adopted here: this project withholds the header, sender and URL fields that
carry the strongest archive markers, so the probe is working from a deliberately
weakened representation and still succeeds.

## 3.4 Leave-one-corpus-out evaluation

The random split cannot distinguish phishing signal from corpus signal, because
every corpus appears in both training and test. Leave-one-corpus-out evaluation
does distinguish them: the model is retrained six times, each time with one
corpus entirely withheld, and evaluated on that unseen corpus
(`notebooks/05_loco.ipynb`, `artifacts/metrics/loco/`).

| Held-out corpus | Rows | Accuracy | Precision | Recall | F1 | Macro-F1 | AUC-ROC |
|-----------------|------|----------|-----------|--------|-----|----------|---------|
| SpamAssasin | 6 001 | 0.9435 | 0.8933 | 0.9340 | 0.9132 | 0.9357 | 0.9854 |
| Ling | 2 859 | 0.9031 | 0.6416 | 0.8952 | 0.7475 | 0.8438 | 0.9614 |
| CEAS_08 | 39 154 | 0.8193 | 0.9045 | 0.7559 | 0.8235 | 0.8192 | 0.9281 |
| Enron | 29 965 | 0.7719 | 0.7032 | 0.8960 | 0.7880 | 0.7706 | 0.8914 |
| Nigerian_Fraud | 3 337 | 0.9835 | 1.0000 | 0.9835 | 0.9917 | n/a | n/a |
| Nazario | 1 563 | 0.5202 | 1.0000 | 0.5202 | 0.6843 | n/a | n/a |

**The four corpora that contain both classes give macro-F1 between 0.7706 and
0.9357, against a headline of 0.9871.** The gap between the random-split result
and the worst honest cross-corpus result is 0.2165 macro-F1, and this gap, not
the headline, is the project's actual finding. It falls in the same range as the
0.7607-0.8994 reported by Bhuiyan & Bhuiyan (2026), which is corroboration from
an independent corpus and pipeline.

Two rows require an explicit caveat rather than a comparison. **Nazario and
Nigerian_Fraud are single-class corpora**: both consist entirely of phishing, so
the held-out fold contains no negatives at all. Their confusion matrices confirm
this, with zero true negatives and zero false positives. AUC-ROC and average
precision are undefined on a single-class set and are recorded as NaN in the
artefacts. Their macro-F1 values of 0.3422 and 0.4958 are arithmetic artefacts
of averaging over an absent class and **must not be read as performance
figures**; quoting them alongside the others would misrepresent the result in
both directions. What these two folds do legitimately report is recall on unseen
phishing: 0.9835 on Nigerian_Fraud and 0.5202 on Nazario.

That Nazario recall is the most serious result in this report. Nazario is a
collection of real-world phishing, and a model that has not seen it during
training detects **barely half of it**, at a threshold selected on validation.
The precision of 1.0 is not reassurance, as it is guaranteed when there are no
negatives to misclassify. Enron shows the complementary failure: 5 360 false
positives against 10 431 true negatives, meaning roughly a third of unseen
legitimate corporate mail is flagged. The model does not generalise to genuinely
unseen mail of either class; it degrades differently depending on which kind is
new.

## 3.5 False positives and what the explanations reveal

The LIME analysis (`notebooks/06_xai.ipynb`,
`artifacts/metrics/xai_stability.json`; Figure 5 shows M1's attributions for the
confidently flagged phishing email) was run on four cases, and the false
positive is the informative one.

On a legitimate message that both models flagged, M0 assigned P(phishing) = 0.8661
and M1 assigned 1.0. The message is a short request from the Ling corpus: "korean
software for macintosh dear sir / madam , would you please send me any
information about korean software for macintosh". The tokens the two models
attended to overlapped strongly (Jaccard 0.7647, 13 shared tokens), and the signs
say what drove the verdict. For M1 the strongest pushes towards phishing were
*madam*, *software*, *sir*, *dear* and *you*. The topical words *korean* and
*macintosh*, and the polite *please*, pushed the other way, towards legitimate.

This is the artefact of section 3.3 made visible at the level of a single email.
The model did not flag this message for anything resembling a phishing
technique. It flagged it for the salutation "dear sir / madam", a formal opening
to an unknown reader, and it did so strongly enough to override the topical
words that correctly pointed to a legitimate request. A message is being
penalised for its courtesy.

That both an independently trained linear model and a CNN converge on the same
tokens indicates the problem is in the data, not in either architecture. The
signal is genuinely present in the pooled corpus; both models find it because it
is there to be found.

Attribution stability was checked by re-running LIME under two seeds. On the
false positive and borderline cases agreement is high (Jaccard 0.875 to 1.0),
but on the confident phishing case M1's explanations agree across seeds at only
0.1538, sharing four tokens of fifteen. Explanations of this model's most
confident predictions are therefore not stable enough to present to an analyst
as a definitive reason, and the desktop application's highlighting should be
read as indicative. Predicted probabilities were identical across seeds
(spread 0.0), so the instability is in the local explanation, not the model.

The explanations were cross-checked with Integrated Gradients (Sundararajan,
Taly & Yan, 2017), computed through Captum (Kokhlikyan et al., 2020) at M1's
embedding layer against an all-padding baseline (`notebooks/06_xai.ipynb`,
`artifacts/metrics/xai_ig_agreement.json`; Figure 6). Unlike LIME it uses the network's own
gradients and has no sampling step, and its attributions sum to the difference
between the email's logit and the baseline's; that completeness property held on
every case, with a largest error of 0.0138. Three results matter.

First, it confirms the false-positive reading with a method built on different
principles: *madam* alone accounts for 75.3% of the logit gap and *sir* for
40.5% (shares exceed 100% in total because other tokens pull the other way),
while *macintosh* and *korean* push towards legitimate, as LIME found. Second,
it puts a number on the corpus artefact inside a single prediction: on the
confidently legitimate Enron email, the token *enron* alone carries 77.5% of the
logit gap. The model is certain that email is legitimate mainly because it names
the archive it came from. Third, the two methods agree where an email is short
(top-k Jaccard 0.7647 and 1.0 on the false positive and borderline cases) and
diverge on long ones (0.2 and 0.1111), which are exactly the cases where M1's
LIME explanations were unstable across seeds (0.1538 on both). On the
1 030-token advance-fee email, of which M1 reads the first 400 tokens, Integrated
Gradients attributes the verdict to *thousand*, *money*, *country*, *dollars*
and *million*, a more plausible account than LIME's *god's*, *vos* and *trust*.
Where the methods share a token they agree on its direction 96.43% of the time
on average. The highlighting in the application therefore stays LIME, as
designed, but its reading on long emails should be treated with the caution the
stability result already implies.

## 3.6 Verdict: is the model effective?

**Within the distribution it was trained on, yes. As a general phishing
detector, not yet.**

The case for effectiveness is the triage task of section 1. At 0.9871 macro-F1
and 0.9925 recall on held-out mail from the same pool, ranking an abuse-mailbox
queue by this model's score would put phishing near the top far more often than
chronological order would, and 48 false negatives against 6 385 true positives
is a defensible miss rate for a system where a human reads everything eventually
and nothing is blocked automatically.

The case against deploying it more ambitiously rests on this project's own
measurements. A probe recovers the source corpus from text alone at 0.9445
accuracy, so part of the headline score is archive recognition. Cross-corpus
macro-F1 falls to between 0.7706 and 0.9357 on corpora containing both classes.
Recall on unseen real-world phishing from the Nazario collection is 0.5202.
A new campaign, from a sender and in a register the pooled corpora do not
contain, is the case the LOCO numbers simulate, and it is the case the model
handles worst. The false-positive analysis shows the model attending to
politeness and corpus vocabulary rather than to technique, which is the mechanism behind
that failure rather than a separate problem.

The honest conclusion is that the headline number measures the benchmark and the
LOCO numbers estimate the deployment. Reporting only the former would have
produced a far more impressive report and a false one. Section 4 takes up what
this implies for machine learning in security operations generally.

## References cited in this section

- Bhuiyan, I. & Bhuiyan, T. 2026. The fragility of phishing detection models:
  evidence from cross-corpus transfer, prevalence shift, artifact learning, and
  evasion risk. *Big Data and Cognitive Computing*, 10(7):211.
- Kokhlikyan, N., Miglani, V., Martin, M., Wang, E., Alsallakh, B., Reynolds, J.,
  Melnikov, A., Kliushkina, N., Araya, C., Yan, S. & Reblitz-Richardson, O. 2020.
  Captum: a unified and generic model interpretability library for PyTorch.
  arXiv:2009.07896.
- Ribeiro, M.T., Singh, S. & Guestrin, C. 2016. "Why should I trust you?":
  explaining the predictions of any classifier. *Proceedings of KDD 2016*.
- Sundararajan, M., Taly, A. & Yan, Q. 2017. Axiomatic attribution for deep
  networks. *Proceedings of the 34th International Conference on Machine
  Learning*. PMLR 70, pp. 3319-3328.
