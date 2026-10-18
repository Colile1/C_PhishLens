# 4. Real-world use of machine learning in cybersecurity

## 4.1 The four application areas

**Intrusion detection.** Network and host intrusion detection is the most
studied application of machine learning in security. Classifiers over flow
records and system call traces detect known attack classes at high reported
accuracy, and the field has moved from signature matching to learned models
precisely because signatures cannot cover variants. Its practical difficulty is
that published benchmarks are dominated by a handful of ageing labelled datasets,
so reported performance reflects those datasets' composition as much as the
methods' merits, a problem this project encountered directly in its own domain
(section 3.3).

**Threat classification.** Assigning a label to an observed artefact: malware
family attribution, spam and phishing classification, and the triage of security
alerts by type. The task this project addresses belongs here. Its strength is
throughput: a model labels at a rate no analyst team can match. Its weakness is
that labels are assigned from learned surface regularities that may not be the
defining property of the class, which is exactly what section 3.5 demonstrated
when the model flagged courteous phrasing rather than any phishing technique.

**Anomaly detection.** Unsupervised and semi-supervised methods model normal
behaviour and flag deviation, which is the standard approach to insider threat
and account compromise where labelled attack examples are scarce or absent. The
strength is coverage of genuinely novel behaviour that no classifier was trained
for. The corresponding weakness is severe and structural: anomalous is not
malicious. Rare-but-benign behaviour is common in real networks, so the
precision of anomaly detectors at operational base rates is typically poor, and
the burden falls on the analysts described in section 4.4.

**Risk assessment and alert prioritisation.** Rather than deciding whether
something is an attack, these systems rank what should be attended to first:
scoring vulnerabilities by exploitation likelihood, ranking assets by exposure,
or ordering an alert queue. The model built in this project is a risk
prioritisation tool in this sense, since its deployed output is an ordering of
a queue rather than a block decision. This framing is the most defensible use of
an imperfect model, because a mis-ranked item is still eventually read, whereas
a wrongly blocked item is simply lost.

## 4.2 Data quality

Security data is expensive to label and noisy once labelled. Ground truth
usually requires expert judgement, and the experts qualified to provide it are
the same scarce analysts the model is meant to relieve. The result is that
public corpora are assembled from archives collected at different times, by
different projects, under different conventions, and pooled afterwards to reach
a usable size.

This project measured the consequence of that pooling rather than assuming it
away. Bhuiyan & Bhuiyan (2026) report that a linear probe recovers the source
corpus of a pooled phishing dataset at 97-98% accuracy; the probe built here
reached 0.9445 accuracy from message text alone, despite deliberately
withholding the header and URL fields that carry the strongest archive markers
(section 3.3). A dataset that identifies its own provenance that reliably
allows a classifier to score well by learning provenance, because provenance and
label are correlated by how the pool was assembled.

The practical consequence is that a headline benchmark figure on a pooled corpus
should be treated as an upper bound obtained under favourable conditions, not as
an estimate of field performance. The corrective is not better preprocessing but
a different evaluation protocol: hold out an entire source, as in the
leave-one-corpus-out results of section 3.4, and report the gap.

## 4.3 Concept drift

Every deployed model faces drift as the world changes. Security is distinguished
by drift being **adversarial and deliberate** rather than incidental. A retail
demand model drifts because tastes change; a phishing model drifts because
attackers observe what is being caught and change it on purpose. Drift in
security is therefore an active response to the defence, and it accelerates
whenever the defence improves.

Shyaa *et al.* (2024), surveying concept drift in intrusion detection,
document both the rate of degradation and the mitigation strategies:
drift detection, periodic retraining, and feature representations chosen for
stability rather than immediate discriminative power.

This project's own data illustrates the problem at its starkest. The pooled
corpora include the Enron collection from the early 2000s and CEAS-08 from 2008,
and a model fitted to them is here being asked about 2026 campaigns. The
vocabulary of a 2008 lure and a 2026 lure differ substantially, and the model's
strongest features are lexical. The LOCO results in section 3.4 are a partial
proxy for temporal drift, since holding out a corpus holds out a collection era
along with it, and cross-era macro-F1 falling to 0.7706 on Enron is consistent
with drift being a first-order effect rather than a slow decay. The operational
implication is that a phishing model requires a retraining pipeline and drift
monitoring from the first day of deployment, and that a model shipped without
one has a shelf life measured in months.

## 4.4 False positives and the alert budget

The most consequential gap between research and operational practice concerns
false positives. Research evaluates on balanced or near-balanced test sets;
operations run at base rates where the negative class overwhelms the positive
one, and the constraint is analyst time rather than accuracy.

The arithmetic is unforgiving. A model at 99% precision applied to a queue of
100 000 messages produces 1 000 false alerts. If each takes an analyst three
minutes to dismiss, that is fifty hours of work generated by a model described
as 99% precise. Because analyst capacity is fixed, every security deployment has
an implicit **alert budget**, and the meaningful question is not the model's
precision but its recall *within* the number of alerts the team can actually
process.

Bhuiyan & Bhuiyan (2026) make this concrete through prevalence shift: models
trained at high phishing prevalence lose recall sharply once realistic
prevalence and an operational alert budget are imposed. A model tuned on a 52%
phishing corpus, as here, is calibrated for a world that does not exist
operationally, where phishing is a small fraction of mail.

This project's own false-positive count illustrates the point. On the random
split M1 produced 111 false positives against 5 829 true negatives, which is
tolerable. On the held-out Enron corpus it produced 5 360 false positives
against 10 431 true negatives (section 3.4). Deployed against a corporate mail
stream resembling Enron's, this model would flag roughly a third of legitimate
mail, and no analyst team could absorb that. The deployment design of section 1
anticipates this by using the model to order a queue rather than to block
anything, which converts a false positive from a destroyed message into a
message read slightly too early.

## 4.5 The research-to-deployment gap

Apruzzese *et al.* (2023) address directly why models that perform well in
publications underperform in security operations centres. The causes are
consistent with everything above: benchmark datasets whose composition does not
match deployment, evaluation metrics that ignore the cost asymmetry between
error types, base rates that differ by orders of magnitude between test set and
production, and the absence of an adversary from the evaluation entirely.

That published scores on pooled phishing corpora are routinely high supports the
argument rather than undermining it. Mendes *et al.* (2025) report 98.34% F1 on a
135 894-email multi-source corpus using text and engineered features, a figure comparable to this
project's own 0.9871 macro-F1. When high scores on this task are ordinary, the
score is not the contribution, and a report that presents one as an achievement
has measured the benchmark rather than the problem.

What operational evaluation would require instead is visible in this project's
structure: hold out an entire source rather than a random sample, test whether
the dataset predicts itself before trusting any classifier trained on it,
evaluate at realistic prevalence under a fixed alert budget, and inspect the
tokens driving individual decisions to confirm the model attends to the
phenomenon rather than to a correlate of it.

## 4.6 What this project's results imply for deploying its own model

Applying that standard to the model built here gives a specific answer rather
than a general caution.

It is suitable for deployment in the narrow role of section 1: ranking a
forwarded abuse-mailbox queue, with every message eventually read by an analyst,
nothing blocked automatically, and the highlighted tokens shown alongside each
score. Within that role its 0.9925 recall on in-distribution mail is genuinely
useful and its errors are recoverable.

It is not suitable for automatic filtering, for deployment against a live
corporate mail stream, or for any workflow where its output reaches a customer
without a person in between. The evidence is this project's own: corpus-probe
accuracy of 0.9445, cross-corpus macro-F1 as low as 0.7706, recall of 0.5202 on
unseen real-world phishing, and explanations showing the model responding to
politeness and corpus vocabulary rather than to phishing technique.

Were it to be deployed, three things would be required before and not after: a
retraining pipeline with drift monitoring, a calibration of the threshold at
operational prevalence against a stated alert budget, and a held-out evaluation
on the target organisation's own mail rather than on a public corpus. The
general conclusion is that a model's readiness for security operations is
established by how it was evaluated rather than by how well it scored, and that
a project able to state where its model fails is more useful to a defender than
one able only to state where it succeeds.

## References cited in this section

- Apruzzese, G., Laskov, P., Montes de Oca, E., Mallouli, W., Búrdalo Rapa, L.,
  Grammatopoulos, A.V. & Di Franco, F. 2023. The role of machine learning in
  cybersecurity. *Digital Threats: Research and Practice*, 4(1).
- Bhuiyan, I. & Bhuiyan, T. 2026. The fragility of phishing detection models.
  *Big Data and Cognitive Computing*, 10(7):211.
- Mendes, P., Maia, E. & Praça, I. 2025. MeAJOR corpus: a multi-source dataset
  for phishing email detection. arXiv:2507.17978. [Preprint.]
- Shyaa, M.A., Ibrahim, N.F., Zainol, Z., Abdullah, R., Anbar, M. &
  Alzubaidi, L. 2024. Evolving cybersecurity frontiers: a comprehensive survey
  on concept drift and feature dynamics aware machine and deep learning in
  intrusion detection systems. *Engineering Applications of Artificial
  Intelligence*, 137(Part A):109143.
