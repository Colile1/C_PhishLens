# src/phishlens/explain/

Token-level explanations : the extra features, and the thing that makes the desktop
app worth using rather than just worth demonstrating. **Correctness-critical**
: a misaligned attribution map produces a confident and false
explanation, which is worse than no explanation at all, and it will be on screen
during the demo.

## Files

| File | Responsibility |
|------|----------------|
| `predict_fn.py` | `make_text_cnn_predict_fn` / `make_baseline_predict_fn` : wrap M1 and M0 in one two-column probability contract (column 1 = phishing). This is the single inference path LIME, the notebook and the `/explain` endpoint all call, so an explanation always describes the deployed pipeline |
| `lime_explainer.py` | `explain_text(text, predict_fn, config, seed=None) -> Explanation` : wrap `LimeTextExplainer`, seeded from config so an explanation is reproducible. Explains the **normalised** text with the project tokeniser as LIME's `split_expression` |
| `attributions.py` | `align_to_tokens(text, weights) -> list[TokenWeight]` : map LIME's per-word-type weights onto every token occurrence in the explained text, in order, with character offsets. Driven by the text's own tokens, never by the weight map : the off-by-one this file exists to prevent. Also `top_tokens`, ranking by absolute weight so evidence for *legitimate* is not hidden |
| `stability.py` | `stability_report(text, predict_fn, config, seeds=None) -> StabilityReport` : explain one email under each seed in `config.explain.stability_seeds` and report the top-k Jaccard overlap, the agreed tokens and the probability spread. Low overlap is a finding for the report, not something to hide |

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

Done when: **met.** Attribution tokens match the tokens
actually present in the input, in order, with no off-by-one, proved on
adversarial fixtures (a repeated token, a token that is a substring of
another, the `<url>`/`<num>` placeholders) by
`tests/critical/test_attribution_alignment.py` and re-asserted on real emails
inside `notebooks/06_xai.ipynb`; a known phishing email attributes to
plausible lure tokens; and the stability report is recorded for four emails
under two models in `artifacts/metrics/xai_stability.json`.
