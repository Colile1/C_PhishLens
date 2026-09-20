# src/phishlens/explain/

Token-level explanations : the extra features, and the thing that makes the desktop
app worth using rather than just worth demonstrating. **Correctness-critical**
: a misaligned attribution map produces a confident and false
explanation, which is worse than no explanation at all, and it will be on screen
during the demo.

## Planned files

| File | Responsibility |
|------|----------------|
| `lime_explainer.py` | `explain_text(text, predict_fn, config) -> Explanation` : wrap `LimeTextExplainer`, return per-token weights with the sign convention documented: positive pushes towards phishing. Seeded from config so an explanation is reproducible |
| `attributions.py` | `align_to_tokens(text, weights) -> list[TokenWeight]` : map LIME's weights back onto the tokens as they appear in the original text, in order, with their character offsets. The off-by-one that this file exists to prevent is the whole reason this step is critical |
| `stability.py` | `stability_report(text, predict_fn, seeds) -> StabilityReport` : explain the same email under each seed in `config.explain.stability_seeds` and report the overlap of the top-k tokens. Low overlap is a finding for the report, not something to hide |
| `integrated_gradients.py` | *Optional.* `explain_with_ig(model, encoded, vocabulary) -> list[TokenWeight]` : Captum cross-check for M1. Only built if step 12 has time; the sign convention must match `lime_explainer` |

## Rules

- The predict function handed to LIME is the **same** path the API uses, not a
  reimplementation. An explanation of a different pipeline is not an explanation.
- Attributions are returned with character offsets, so the desktop app highlights
  the exact span in the user's own text rather than a re-tokenised copy of it.
- The sign convention is stated once, here, and every consumer follows it.
- An explanation is local. The report and the app must both say so : these
  weights describe this prediction, not the model in general (Ribeiro et al., 2016).

## Integration

Imports `config`, `features` and `models`. Imported by the XAI notebook and by
`serving` (the `/explain` endpoint). Must not import `serving` or `desktop`.

Done when: attribution tokens match the tokens actually present in the input, in
order, with no off-by-one, proved by a test on a known string; a known phishing
email attributes to plausible lure tokens; and the stability report is recorded
for at least three emails.
