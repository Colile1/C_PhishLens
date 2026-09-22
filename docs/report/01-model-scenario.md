# 1. Model scenario (3 points)

*How is this model useful in the context of cybersecurity?*

## The operational setting

The scenario this model is built for is the abuse mailbox of a South African
retail bank. Customers forward messages they distrust, staff forward messages
that survived the mail gateway, and the resulting queue is read by a small
security operations team. The queue is larger than the team. On any given day
the analysts cannot read every message, so the order in which messages are read
determines which phishing campaign is discovered in the first hour and which is
discovered after the losses are already reported.

The model sits in front of that queue. Each forwarded message is scored from its
subject and body, and the queue is ordered by that score so the most likely
phishing is read first. Alongside the score the model returns the tokens that
drove it, which the analyst sees as highlighted words in the desktop client.
The analyst therefore reads a decision and a reason together, and can overrule
a wrong score in seconds rather than reconstructing the model's logic from
nothing.

## Why a learned model rather than rules

Gateway filtering and signature matching remain necessary and are not replaced
here. Their weakness is specific: both recognise campaigns that have been seen
before. A signature is written after a campaign is identified, and a blocklist
entry is added after a domain is reported. A newly registered domain sending a
freshly worded lure matches neither, and it is precisely those messages that
reach the abuse mailbox, because the mailbox is where the gateway's misses
accumulate. A model that scores the language of a message can rank an unseen
campaign above routine noise on the strength of its phrasing, before any
signature exists for it.

## Why text alone is a deliberate constraint

The model reads subject and body text only. Sender domain, header residue, URL
counts and routing metadata are excluded by design, although the raw corpora
carry them. Those fields are the strongest predictors available on this data
and excluding them lowers the headline score, which is the point. Header and
domain features on a pooled research corpus encode which archive a message was
drawn from at least as much as they encode phishing, and a model permitted to
use them can score well without learning anything about phishing language at
all. Section 3 measures how much of that artefact survives even under the text-
only constraint. Restricting the input is what makes the resulting claim about
phishing language an honest one, and it is also what makes the token-level
explanation meaningful to an analyst: highlighted words in the message body are
something a person can judge, whereas a weight on a header field is not.

## What the model is not

Three boundaries define the deployment and make the false-positive discussion in
section 3 coherent.

It is not an automatic blocker. Nothing is deleted, quarantined or bounced on
the model's output. The output is an ordering and an explanation, and the
consequence of a wrong score is a message read in the wrong order, not a
customer's legitimate mail destroyed.

It is not a replacement for the mail gateway. It operates on messages that have
already been forwarded by a human who found them suspicious, so its input
distribution is a triage queue rather than the full mail stream.

It does not act without a person. Every decision that reaches a customer is made
by an analyst who has seen both the score and the tokens behind it. This is the
condition under which a model with the generalisation limits reported in section
3 can be deployed responsibly at all: the model is allowed to be wrong, because
the workflow assumes it will be, and the cost of each error is bounded by a
person reading the message anyway.
