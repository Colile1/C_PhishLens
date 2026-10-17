// tokens.js - render LIME token weights as shaded text and rank the
// strongest signals. Rendering only: no HTTP.
// Author: Colile Sibanda

// Five steps per direction: enough to read intensity without implying a
// precision the weights do not have. Matches src/desktop/highlight.py.
const SHADES = 5;

/**
 * Pick the CSS classes for one weight.
 * Inputs: weight - signed attribution, positive means towards phishing;
 * strongest - the largest absolute weight in this explanation.
 * Output: a class string such as "tok-p s3", or "" when unshaded.
 */
function shadeClass(weight, strongest) {
  if (weight === 0 || strongest === 0) return "";
  const step = Math.min(Math.floor((Math.abs(weight) / strongest) * SHADES), SHADES - 1) + 1;
  return `${weight > 0 ? "tok-p" : "tok-l"} s${step}`;
}

/**
 * Show `text` with each token shaded by its weight.
 * `text` must be the normalised string the API returned: the token
 * offsets index into it, not into the raw input.
 * Inputs: container - element to fill; text - the explained string;
 * tokens - [{token, weight, start, end}] in order.
 * Output: none; replaces the container's children.
 */
function renderTokens(container, text, tokens) {
  clearChildren(container);
  const strongest = tokens.reduce((m, t) => Math.max(m, Math.abs(t.weight)), 0);
  const ordered = [...tokens].sort((a, b) => a.start - b.start);
  let cursor = 0;
  for (const t of ordered) {
    const cls = shadeClass(t.weight, strongest);
    if (!cls || t.start < cursor) continue;
    container.appendChild(document.createTextNode(text.slice(cursor, t.start)));
    const span = document.createElement("span");
    span.className = cls;
    span.textContent = text.slice(t.start, t.end);
    const direction = t.weight > 0 ? "towards phishing" : "towards legitimate";
    span.title = `${t.token}: weight ${fmt(t.weight, 4)}, ${direction}`;
    container.appendChild(span);
    cursor = t.end;
  }
  container.appendChild(document.createTextNode(text.slice(cursor)));
}

/**
 * Rank the tokens that pushed hardest each way.
 * A repeated token is listed once, under its largest absolute weight,
 * compared case-insensitively. Zero weights are dropped.
 * Inputs: tokens - [{token, weight}]; limit - rows per direction.
 * Output: {phishing: [...], legitimate: [...]}, strongest first.
 */
function topSignals(tokens, limit = 5) {
  const best = new Map();
  for (const t of tokens) {
    if (t.weight === 0) continue;
    const key = t.token.toLowerCase();
    const seen = best.get(key);
    if (!seen || Math.abs(t.weight) > Math.abs(seen.weight)) best.set(key, t);
  }
  const all = [...best.values()].sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));
  return {
    phishing: all.filter((t) => t.weight > 0).slice(0, limit),
    legitimate: all.filter((t) => t.weight < 0).slice(0, limit),
  };
}
