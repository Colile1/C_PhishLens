# References : C_PhishLens

Working reference list for the report, mainly theory §3.4 (real-world use of ML
in cybersecurity, 7 points) and §3.2 (architecture).

**Verification status** matters. A reference marked *verified* has had its page
opened and its claims read. A reference marked *confirm* is real but the author
list, page numbers or article number still need checking against the publisher
page before it goes into the report. Do not submit with anything still on
*confirm*.

submissions.

---

## Core : the artefact and generalisation argument

**R1 : verified.**
Bhuiyan, I. & Bhuiyan, T. 2026. The fragility of phishing detection models:
evidence from cross-corpus transfer, prevalence shift, artifact learning, and
evasion risk. *Big Data and Cognitive Computing*, 10(7):211.
https://doi.org/10.3390/bdcc10070211

*Why it matters here:* this is the backbone of the project's honesty argument.
On a pooled phishing corpus, a source-corpus probe reached 0.9722 accuracy with
logistic regression and 0.9806 with a linear SVC, meaning models learn corpus
markers, date tokens, URL and domain terms and header residue rather than
general phishing signal. Leave-one-corpus-out F1 fell to 0.8201–0.8994 for
logistic regression and 0.7607–0.8910 for linear SVC. Also documents prevalence
shift: models trained at high phishing prevalence lose recall sharply once an
operational alert budget is imposed. Cite in §3.3 and §3.4, and as the
justification.

---

## Real-world deployment, drift and false positives (§3.4)

**R2 : confirm author list and article number.**
Apruzzese, G., Laskov, P., Montes de Oca, E., Mallouli, W., Búrdalo Rapa, L.,
Grammatopoulos, A.V. & Di Franco, F. 2023. The role of machine learning in
cybersecurity. *Digital Threats: Research and Practice*, 4(1).
https://doi.org/10.1145/3545574

*Why it matters here:* the standard reference for the gap between reported
research performance and operational deployment : why a model that scores well
in a paper underperforms in a SOC. Use for the strengths and limitations part
of §3.4.

**R3 : confirm author list.**
2024. Evolving cybersecurity frontiers: a comprehensive survey on concept drift
and feature dynamics aware machine and deep learning in intrusion detection
systems. *Engineering Applications of Artificial Intelligence*, 137:109143.
https://doi.org/10.1016/j.engappai.2024.109143

*Why it matters here:* concept drift is one of the three challenges the brief
names explicitly. Phishing drifts fast : a model trained on 2008 CEAS mail is
being asked about 2026 campaigns. Use in §3.4 and to frame the temporal
limitation of this dataset.

**R4 : confirm author list and venue details.**
MeAJOR corpus: a multi-source dataset for phishing email detection.
arXiv:2507.17978. https://arxiv.org/abs/2507.17978

*Why it matters here:* a recent multi-source phishing corpus of 135 894 emails
reporting 98.34% F1 with XGBoost on text plus URL features. Useful as the
comparison point for "what do published results on this kind of data look
like", and as evidence that high scores on pooled corpora are routine : which
strengthens the argument that the score alone is not the contribution.

---

## Architecture (§3.2)

**R5 : confirm page numbers.**
Kim, Y. 2014. Convolutional neural networks for sentence classification.
*Proceedings of the 2014 Conference on Empirical Methods in Natural Language
Processing (EMNLP)*. Doha: Association for Computational Linguistics.
https://doi.org/10.3115/v1/D14-1181

*Why it matters here:* the architecture M1 implements. Cite when justifying
parallel convolutions of width 3, 4 and 5 with max-over-time pooling as
detectors of local phrases, which is what phishing lures are.

**R6 : confirm.**
Sanh, V., Debut, L., Chaumond, J. & Wolf, T. 2019. DistilBERT, a distilled
version of BERT: smaller, faster, cheaper and lighter. arXiv:1910.01108.
https://arxiv.org/abs/1910.01108

*Only needed if M2 is attempted.*

---

## Explainability (§3.3 and the bonus)

**R7 : confirm page numbers.**
Ribeiro, M.T., Singh, S. & Guestrin, C. 2016. "Why should I trust you?":
explaining the predictions of any classifier. *Proceedings of the 22nd ACM
SIGKDD International Conference on Knowledge Discovery and Data Mining*. San
Francisco: ACM. https://doi.org/10.1145/2939672.2939778

*Why it matters here:* LIME, the primary attribution method. Cite when
describing what the highlighted tokens in the desktop app actually represent —
a local linear approximation, not a global statement about the model.

**R8 : confirm.**
Lundberg, S.M. & Lee, S.-I. 2017. A unified approach to interpreting model
predictions. *Advances in Neural Information Processing Systems 30 (NeurIPS)*.
https://arxiv.org/abs/1705.07874

*Only needed if SHAP is used as the cross-check instead of Captum.*

---

## To find before writing §3.4

The brief asks for intrusion detection, threat classification, anomaly
detection **and** risk assessment. R1–R4 cover phishing, drift and deployment.
Still needed:

- [ ] One reference on ML-based anomaly detection in production and its false-positive burden
- [ ] One reference on ML in security risk assessment or alert prioritisation
- [ ] One South African or African-context source if one exists, since the report's scenario is a South African financial institution

Add them here with the same verification marking before citing.
