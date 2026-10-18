# 2. Model architecture

The deliverable model, M1, is a Text-CNN implemented in PyTorch, following the
sentence-classification architecture of Kim (2014). A TF-IDF and logistic
regression model, M0, is trained on the same splits as a reference point.
Implementation: `src/phishlens/models/text_cnn.py`, printed in full inside
`notebooks/03_train_cnn.ipynb` with `inspect.getsource`.

## 2.1 Input representation

Subject and body are concatenated with an explicit `<sep>` marker so the model
can distinguish the two fields without being given them as separate inputs.
The concatenated text is lowercased, stripped of HTML, and then three classes of
token are replaced with placeholders: URLs become `<url>`, email addresses
become `<email>`, and digit runs become `<num>`.

Placeholder substitution is a modelling decision rather than a cleaning step. A
specific URL or account number appears in one campaign and never again, so
leaving it in the vocabulary produces a feature that fires on exactly one
training email and cannot generalise. Replacing it retains the fact that a link
was present, which is a genuine property of the message, while discarding the
identity of the link, which is not a property the model should learn from a
corpus this small.

The text is then integer encoded against a vocabulary of 40 000 tokens with a
minimum frequency of 2, and padded or truncated to 400 tokens. The vocabulary
is fitted on the training split only. Fitting it on the pooled data before
splitting would let test-set words influence the training representation, which
is a quiet form of leakage that does not show up in any metric.

## 2.2 Layers

| Layer | Configuration | Output shape |
|-------|---------------|--------------|
| Embedding | 40 000 × 128, `padding_idx=0` | (batch, 400, 128) |
| Conv1d × 3 | widths 3, 4, 5; 100 filters each; ReLU | (batch, 100, 400−k+1) |
| Max-over-time pooling | max over the time axis | (batch, 100) each |
| Concatenation | three pooled vectors | (batch, 300) |
| Dropout | p = 0.5 | (batch, 300) |
| Linear | 300 → 1 | (batch,) |

Trainable parameters total 5 274 201, of which 5 120 000 are the embedding
table. The convolutional and classification layers together account for 154 201
parameters, so the model's capacity is overwhelmingly lexical.

**Embedding.** Embeddings are learned from scratch at 128 dimensions rather than
initialised from pre-trained vectors. Pre-trained vectors are trained on prose
and carry no representation for the placeholder tokens that matter most here,
and the training corpus of 57 736 emails is large enough to fit 128 dimensions
directly. The padding index is pinned to a zero vector so that padding
contributes nothing to the pooled features, which matters because most emails
are shorter than 400 tokens and padding would otherwise dominate the input.

**Convolutional block.** Three convolutions run in parallel over the embedded
sequence with widths of 3, 4 and 5 tokens, 100 filters each, followed by ReLU.
This is the architectural argument for the whole model. A filter of width 4
sliding over word embeddings is a detector for four-word phrases, and phishing
lures are phrases: "verify your account immediately", "your payment was
declined", "confirm your banking details". The signal this task depends on is
local and lexical, so an architecture whose inductive bias is local phrase
detection matches the structure of the data rather than fighting it. Running
three widths in parallel avoids committing to a single phrase length.

**Max-over-time pooling.** Each filter's activation across the whole sequence is
reduced to its single maximum. The consequence is positional indifference: a
lure buried at the end of a long quoted reply chain produces the same pooled
activation as the same lure in the opening line. Emails vary enormously in
length and the placement of the lure within them is arbitrary, so a
representation that asks only whether a phrase occurred, not where, is the
correct one. It also yields a fixed 300-dimensional vector regardless of input
length.

**Dropout and output.** Dropout at p = 0.5 is applied to the pooled features,
which is the point of highest overfitting risk since 300 features feed a single
linear layer. The output is one raw logit rather than a two-class softmax. The
logit is paired with `BCEWithLogitsLoss`, which applies the sigmoid internally
for numerical stability, and it keeps the decision threshold an explicit,
reportable quantity instead of hiding it inside an `argmax` at 0.5.

**Optimisation.** Adam at a learning rate of 0.001, batch size 64.

## 2.3 Class weighting

The pooled corpus is 52% phishing, a mild imbalance. It is handled with a
positive class weight in the loss rather than by resampling. Oversampling the
minority class would duplicate emails, and because de-duplication and splitting
in this project operate on content-hash groups specifically to keep
near-identical emails out of two splits at once, deliberately reintroducing
duplicates into training would work against that guarantee. Undersampling would
discard real emails from a corpus whose per-corpus counts are already uneven.
Weighting the loss changes the gradient without changing the data.

## 2.4 Training loop, patience and best-weight restore

The loop is written explicitly in `src/phishlens/training/loop.py` rather than
delegated to a framework callback, and is printed into
`notebooks/03_train_cnn.ipynb`. After every epoch the model is evaluated on the
validation split and a full metric row is appended to
`artifacts/metrics/training_log.csv`: loss, accuracy, precision, recall, F1,
macro-F1 and AUC-ROC, for both train and validation.

Early stopping watches **validation macro-F1**, not validation loss, with
`patience = 3` and `min_delta = 0.001`. Macro-F1 is the quantity the project is
trying to maximise, and loss and macro-F1 do not always move together; stopping
on loss can therefore discard an epoch that was better at the actual task.
Whenever validation macro-F1 improves by more than `min_delta`, the model's
state dictionary is deep-copied as the best checkpoint. When three consecutive
epochs fail to improve on it, training halts and **the best checkpoint is
restored into the model before any evaluation runs**.

This restore is the guarantee behind every number in section 3. Training ran for
six epochs and the best validation macro-F1 was reached at epoch 6 (0.98665,
against 0.98624 at epoch 4). The reported test metrics come from the restored
best epoch, not from whichever epoch training happened to end on. Reporting last-
epoch numbers after early stopping is a common and silent error: it reports a
model that was deliberately rejected.

The test split is read exactly once, by the reporting notebook, after all of
this is complete. The decision threshold is selected on validation
(0.5698 for M1, 0.5055 for M0) and then applied unchanged to test.

## 2.5 Why this architecture rather than an LSTM or a transformer

An LSTM models long-range sequential dependency. That is the wrong prior for
this data: phishing signal is a phrase, not a discourse structure spanning
hundreds of tokens, and an LSTM must process 400 tokens sequentially while the
CNN's three convolutions run in parallel. The extra capacity would be spent
modelling a dependency the task does not contain.

A fine-tuned transformer such as DistilBERT (Sanh *et al.*, 2019) would very
likely score higher, and was specified as an optional stretch model, M2. It was
not attempted. The reason is stated plainly rather than presented as a
preference: M1 at 5.3 million parameters trains to convergence on CPU in six
epochs, whereas DistilBERT's 66 million parameters do not, and the results in
section 3 show that the binding constraint on this project is not headline
accuracy. M1 already reaches 0.9871 macro-F1 on the random-split test set. The
gap that matters is the one between that number and the leave-one-corpus-out
numbers, and a larger model fitted to the same pooled corpus would be subject to
the same artefact problem. Spending the remaining effort on measuring that gap
was the better use of it.

## References cited in this section

- Kim, Y. 2014. Convolutional neural networks for sentence classification.
  *Proceedings of EMNLP 2014*. Doha: ACL.
- Sanh, V., Debut, L., Chaumond, J. & Wolf, T. 2019. DistilBERT, a distilled
  version of BERT. arXiv:1910.01108.
