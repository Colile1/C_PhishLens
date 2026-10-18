# References : C_PhishLens

Every work cited in the report (`docs/report/`), in Harvard style, with a note
on what it supports. Each entry was checked against its publisher or archive
page (authors, year, venue, pages or article number) before being cited.

---

## The artefact and generalisation argument

Bhuiyan, I. & Bhuiyan, T. 2026. The fragility of phishing detection models:
evidence from cross-corpus transfer, prevalence shift, artifact learning, and
evasion risk. *Big Data and Cognitive Computing*, 10(7):211.
https://doi.org/10.3390/bdcc10070211

*Used for:* the project's central caution. On a pooled phishing corpus, a
source-corpus probe reached 0.9722 accuracy with logistic regression and 0.9806
with a linear SVC: models learn corpus markers, date tokens, URL and domain terms
and header residue rather than general phishing signal. Leave-one-corpus-out F1
fell to 0.8201–0.8994 for logistic regression and 0.7607–0.8910 for linear SVC.
It also documents prevalence shift. Cited in sections 3 and 4, and the reason
this project measures a corpus probe and a leave-one-corpus-out evaluation.

---

## Real-world deployment, drift and false positives

Apruzzese, G., Laskov, P., Montes de Oca, E., Mallouli, W., Búrdalo Rapa, L.,
Grammatopoulos, A.V. & Di Franco, F. 2023. The role of machine learning in
cybersecurity. *Digital Threats: Research and Practice*, 4(1):article 8, 38 pp.
https://doi.org/10.1145/3545574

*Used for:* the gap between reported research performance and operational
deployment, and why a model that scores well in a paper underperforms in a
security operations centre. Section 4.

Shyaa, M.A., Ibrahim, N.F., Zainol, Z., Abdullah, R., Anbar, M. & Alzubaidi, L.
2024. Evolving cybersecurity frontiers: a comprehensive survey on concept drift
and feature dynamics aware machine and deep learning in intrusion detection
systems. *Engineering Applications of Artificial Intelligence*,
137(Part A):109143. https://doi.org/10.1016/j.engappai.2024.109143

*Used for:* concept drift in intrusion detection and phishing, and the temporal
limitation of training on 2008-era mail. Section 4.

Mendes, P., Maia, E. & Praça, I. 2025. MeAJOR corpus: a multi-source dataset
for phishing email detection. arXiv:2507.17978 (preprint).
https://arxiv.org/abs/2507.17978

*Used for:* a recent multi-source phishing corpus of 135 894 emails reporting
98.34% F1 with XGBoost on text plus URL features, as evidence that high scores on
pooled corpora are routine. Section 4.

---

## Architecture

Kim, Y. 2014. Convolutional neural networks for sentence classification.
*Proceedings of the 2014 Conference on Empirical Methods in Natural Language
Processing (EMNLP)*. Doha: Association for Computational Linguistics.
https://doi.org/10.3115/v1/D14-1181

*Used for:* the architecture M1 implements: parallel convolutions of width 3, 4
and 5 with max-over-time pooling as detectors of local phrases. Section 2.

Sanh, V., Debut, L., Chaumond, J. & Wolf, T. 2019. DistilBERT, a distilled
version of BERT: smaller, faster, cheaper and lighter. arXiv:1910.01108.
https://arxiv.org/abs/1910.01108

*Used for:* the transformer alternative considered and not used. Section 2.

---

## Explainability

Ribeiro, M.T., Singh, S. & Guestrin, C. 2016. "Why should I trust you?":
explaining the predictions of any classifier. *Proceedings of the 22nd ACM
SIGKDD International Conference on Knowledge Discovery and Data Mining*. San
Francisco: ACM, pp. 1135-1144. https://doi.org/10.1145/2939672.2939778

*Used for:* LIME, the primary attribution method, and what the highlighted
tokens represent: a local linear approximation of one prediction, not a global
statement about the model. Section 3.

Sundararajan, M., Taly, A. & Yan, Q. 2017. Axiomatic attribution for deep
networks. *Proceedings of the 34th International Conference on Machine
Learning*. PMLR 70, pp. 3319-3328.
https://proceedings.mlr.press/v70/sundararajan17a.html

*Used for:* Integrated Gradients, the cross-check for M1: the baseline, the path
integral and the completeness property the implementation checks. Section 3.

Kokhlikyan, N., Miglani, V., Martin, M., Wang, E., Alsallakh, B., Reynolds, J.,
Melnikov, A., Kliushkina, N., Araya, C., Yan, S. & Reblitz-Richardson, O. 2020.
Captum: a unified and generic model interpretability library for PyTorch.
arXiv:2009.07896. https://arxiv.org/abs/2009.07896

*Used for:* the library the Integrated Gradients implementation uses
(`LayerIntegratedGradients`). Section 3.

---

## Scope note

Section 4 covers anomaly detection and risk assessment as application areas,
argued from the general properties of those methods (base rates, precision under
class imbalance, ranking rather than blocking) and tied to this project's own
measured results, supported by Apruzzese et al. (2023) and Shyaa et al. (2024).
The South African bank abuse mailbox in section 1 is a scenario framing, not an
empirical claim about the South African threat landscape, so it carries no
citation.
