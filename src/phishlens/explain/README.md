# src/phishlens/explain/

Token-level explanations: the reason a flagged email can be checked rather
than taken on trust. This layer is **correctness-critical**: a misaligned
attribution map produces a confident and false explanation, which is worse than
none.

## Files

| File | Responsibility |
|------|----------------|
| `predict_fn.py` | `make_text_cnn_predict_fn` / `make_baseline_predict_fn` : wrap M1 and M0 in one two-column probability contract (column 1 = phishing). This is the single inference path LIME, the notebook and the `/explain` endpoint all call, so an explanation always describes the deployed pipeline |
| `lime_explainer.py` | `explain_text(text, predict_fn, config, seed=None) -> Explanation` : wrap `LimeTextExplainer`, seeded from config so an explanation is reproducible. Explains the **normalised** text, with the project tokeniser as LIME's `split_expression`, because that is the string the model actually sees |
| `attributions.py` | `align_to_tokens(text, weights) -> list[TokenWeight]` : map LIME's per-word weights onto every token occurrence in the explained text, in order, with character offsets. Driven by the text's own tokens, never by the weight map, which is what prevents an off-by-one. Also `top_tokens`, ranking by absolute weight so evidence for *legitimate* is not hidden |
| `stability.py` | `stability_report(text, predict_fn, config, seeds=None)` : explain one email under each seed in `config.explain.stability_seeds` and report the top-k Jaccard overlap, the agreed tokens and the probability spread |
| `integrated_gradients.py` | `integrated_gradients(text, model, vocabulary, config) -> IGExplanation` : Captum `LayerIntegratedGradients` on M1's embedding layer against the all-`<pad>` baseline. Same sign convention and the same offsets as LIME. Refuses a result that leaks attribution onto padding or fails completeness. Not exported from `__init__`, so the API does not import Captum |
| `method_agreement.py` | `method_agreement(lime, ig) -> MethodAgreement` : top-k Jaccard and sign agreement between LIME and Integrated Gradients for the same email, compared per word type |

## Rules

- The predict function handed to LIME is the **same** path the API uses, not a
  reimplementation.
- Attributions carry character offsets into the normalised text, so a client
  highlights exactly the span the model scored.
- Sign convention, stated once: a **positive** weight pushes towards phishing.
  Every consumer follows it and nothing re-signs.
- An explanation is local (Ribeiro et al., 2016): the weights describe one
  prediction, not the model in general.

## Integration

Imports `config`, `features` and `models`. Imported by `notebooks/06_xai.ipynb`
and by `serving` (the `/explain` endpoint). Must not import `serving` or
`desktop`.

Verified by `tests/critical/test_attribution_alignment.py` (a repeated token, a
token that is a substring of another, the `<url>`/`<num>` placeholders) and
`tests/critical/test_ig_alignment.py` (a planted-token model where every other
token must score exactly zero), and re-asserted on real emails in
`notebooks/06_xai.ipynb`.
