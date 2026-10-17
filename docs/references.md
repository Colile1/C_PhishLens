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

**R2 : verified.** (2026-09-22: author list, volume, issue and article number
checked against the ACM Digital Library record.)
Apruzzese, G., Laskov, P., Montes de Oca, E., Mallouli, W., Búrdalo Rapa, L.,
Grammatopoulos, A.V. & Di Franco, F. 2023. The role of machine learning in
cybersecurity. *Digital Threats: Research and Practice*, 4(1):article 8, 38 pp.
https://doi.org/10.1145/3545574

*Why it matters here:* the standard reference for the gap between reported
research performance and operational deployment : why a model that scores well
in a paper underperforms in a SOC. Use for the strengths and limitations part
of §3.4.

**R3 : verified.** (2026-09-22: author list and article number confirmed.)
Shyaa, M.A., Ibrahim, N.F., Zainol, Z., Abdullah, R., Anbar, M. & Alzubaidi, L.
2024. Evolving cybersecurity frontiers: a comprehensive survey on concept drift
and feature dynamics aware machine and deep learning in intrusion detection
systems. *Engineering Applications of Artificial Intelligence*,
137(Part A):109143. https://doi.org/10.1016/j.engappai.2024.109143

*Why it matters here:* concept drift is one of the three challenges the brief
names explicitly. Phishing drifts fast : a model trained on 2008 CEAS mail is
being asked about 2026 campaigns. Use in §3.4 and to frame the temporal
limitation of this dataset.

**R4 : verified.** (2026-09-22: authors and submission year confirmed on arXiv.
Note the year is 2025, not 2026, and it is a preprint : cite it as such.)
Mendes, P., Maia, E. & Praça, I. 2025. MeAJOR corpus: a multi-source dataset
for phishing email detection. arXiv:2507.17978.
https://arxiv.org/abs/2507.17978

*Why it matters here:* a recent multi-source phishing corpus of 135 894 emails
reporting 98.34% F1 with XGBoost on text plus URL features. Useful as the
comparison point for "what do published results on this kind of data look
like", and as evidence that high scores on pooled corpora are routine : which
strengthens the argument that the score alone is not the contribution.

---

## Architecture (§3.2)

**R5 : verified.** (2026-09-22: pages confirmed via the ACL Anthology.)
Kim, Y. 2014. Convolutional neural networks for sentence classification.
*Proceedings of the 2014 Conference on Empirical Methods in Natural Language
Processing (EMNLP)*. Doha: Association for Computational Linguistics.
https://doi.org/10.3115/v1/D14-1181

*Why it matters here:* the architecture M1 implements. Cite when justifying
parallel convolutions of width 3, 4 and 5 with max-over-time pooling as
detectors of local phrases, which is what phishing lures are.

**R6 : verified.** (2026-09-22: title, authors and year confirmed on arXiv.)
Sanh, V., Debut, L., Chaumond, J. & Wolf, T. 2019. DistilBERT, a distilled
version of BERT: smaller, faster, cheaper and lighter. arXiv:1910.01108.
https://arxiv.org/abs/1910.01108

*Only needed if M2 is attempted.*

---

## Explainability (§3.3 and the bonus)

**R7 : verified.** (2026-09-22: pages confirmed against the ACM record.)
Ribeiro, M.T., Singh, S. & Guestrin, C. 2016. "Why should I trust you?":
explaining the predictions of any classifier. *Proceedings of the 22nd ACM
SIGKDD International Conference on Knowledge Discovery and Data Mining*. San
Francisco: ACM, pp. 1135-1144. https://doi.org/10.1145/2939672.2939778

*Why it matters here:* LIME, the primary attribution method. Cite when
describing what the highlighted tokens in the desktop app actually represent :
a local linear approximation, not a global statement about the model.

**R8 : not cited.** SHAP was not used : Captum was specified as the optional
cross-check and the LIME stability analysis served the purpose instead. Left
here for completeness; verify before citing if that changes.
Lundberg, S.M. & Lee, S.-I. 2017. A unified approach to interpreting model
predictions. *Advances in Neural Information Processing Systems 30 (NeurIPS)*.
https://arxiv.org/abs/1705.07874

*Only needed if SHAP is used as the cross-check instead of Captum.*

**R9 : verified.** (2026-10-18: title, authors, volume and pages confirmed on
the PMLR proceedings page.)
Sundararajan, M., Taly, A. & Yan, Q. 2017. Axiomatic attribution for deep
networks. *Proceedings of the 34th International Conference on Machine
Learning*. PMLR 70, pp. 3319-3328.
https://proceedings.mlr.press/v70/sundararajan17a.html

*Why it matters here:* Integrated Gradients, the M1 cross-check. Cite for the baseline, the path integral and the completeness property
the implementation checks.

**R10 : verified.** (2026-10-18: title and author list confirmed on arXiv.)
Kokhlikyan, N., Miglani, V., Martin, M., Wang, E., Alsallakh, B., Reynolds, J.,
Melnikov, A., Kliushkina, N., Araya, C., Yan, S. & Reblitz-Richardson, O. 2020.
Captum: a unified and generic model interpretability library for PyTorch.
arXiv:2009.07896. https://arxiv.org/abs/2009.07896

*Why it matters here:* the library the Integrated Gradients implementation
uses (`LayerIntegratedGradients`).

---

## Status of the §3.4 reference gap (closed 2026-09-22, the build)

The brief asks §3.4 to cover intrusion detection, threat classification,
anomaly detection **and** risk assessment. The earlier plan was to add a
dedicated reference for each of the last two plus a South African source.
Resolution as drafted:

- **Anomaly detection and risk assessment** are covered in §4.1 as application
  areas, argued from the general properties of those methods (base rates,
  precision under class imbalance, ranking rather than blocking) and tied back
  to this project's own measured results. They are supported by R2, which
  covers the deployment gap across application areas, and R3, which covers
  intrusion detection and drift directly. No claim in §4.1 rests on a source
  not in this list.
- **A South African or African-context source** was searched for and none
  suitable was found. The report therefore uses the South African bank abuse
  mailbox as a *scenario framing* in §1 rather than as an empirical claim about
  the South African threat landscape, so no citation is owed. Do not add a
  weak or unverified local source merely to fill this slot.

Every reference R1-R7 that the report cites is marked verified above. R8 is
listed but not cited. Nothing is outstanding on *confirm*.
