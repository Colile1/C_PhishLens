// analyse.js - Analyse page: send the email to /explain and show the
// verdict, the probability against the threshold, and the token
// highlights. Author: Colile Sibanda

const EXAMPLE_EMAIL =
  "URGENT: your account has been suspended\n\n" +
  "Verify your password immediately at http://secure-update.example/login " +
  "or your account will be closed within 24 hours. Click here to confirm " +
  "your identity now.";

/** Format a probability as a percentage without rounding to 0% or 100%. */
function percent(p) {
  if (p < 0.001) return "under 0.1%";
  if (p > 0.999) return "over 99.9%";
  return `${(p * 100).toFixed(1)}%`;
}

/** Fill the verdict line and the probability meter from an /explain body. */
function renderVerdict(result) {
  const verdict = document.getElementById("verdict");
  const meter = document.getElementById("meter");
  clearChildren(verdict);
  clearChildren(meter);

  const phishing = result.label === "phishing";
  const line = document.createElement("p");
  line.className = phishing ? "verdict phishing" : "verdict positive";
  const side = phishing ? "at or above" : "below";
  line.textContent =
    `${phishing ? "Likely phishing" : "Likely legitimate"}: the model gives this email ` +
    `${percent(result.probability)} probability of being phishing, ` +
    `${side} the decision threshold of ${fmt(result.threshold, 3)}.`;
  verdict.appendChild(line);

  const bar = document.createElement("div");
  bar.className = "meter";
  bar.setAttribute("role", "img");
  bar.setAttribute("aria-label", `Probability ${fmt(result.probability, 3)}, threshold ${fmt(result.threshold, 3)}`);
  const fill = document.createElement("span");
  fill.className = "meter-fill";
  fill.style.width = `${result.probability * 100}%`;
  const mark = document.createElement("span");
  mark.className = "meter-mark";
  mark.style.left = `${result.threshold * 100}%`;
  bar.append(fill, mark);
  meter.append(
    bar,
    paragraph(`Bar: probability of phishing (${fmt(result.probability, 3)}), from 0 at the left to 1 at the right. ` +
      `Black line: decision threshold (${fmt(result.threshold, 3)}), chosen on validation data.`, "hint"),
  );
}

/** Fill the strongest-signals table from the token list. */
function renderSignals(tokens) {
  const wrap = document.getElementById("signals");
  clearChildren(wrap);
  const { phishing, legitimate } = topSignals(tokens);
  const rows = [
    ...phishing.map((t) => ({ word: t.token, direction: "towards phishing", weight: fmt(t.weight, 4) })),
    ...legitimate.map((t) => ({ word: t.token, direction: "towards legitimate", weight: fmt(t.weight, 4) })),
  ];
  const table = buildTable(rows);
  if (table) wrap.appendChild(table);
  else wrap.appendChild(paragraph("No word carried any weight in this explanation.", "hint"));
}

/** Show or hide the three result cards together. */
function showResults(visible) {
  ["result-card", "why-card", "signals-card"].forEach((id) => {
    document.getElementById(id).hidden = !visible;
  });
}

document.addEventListener("DOMContentLoaded", () => {
  const input = document.getElementById("email");
  const button = document.getElementById("analyse");
  const banner = document.getElementById("page-banner");
  const count = document.getElementById("count");

  const updateCount = () => {
    const text = input.value.trim();
    const words = text ? text.split(/\s+/).length : 0;
    count.textContent = `${words} words, ${text.length} characters`;
  };

  async function analyse() {
    if (!input.value.trim()) {
      showBanner(banner, "Enter some email text before clicking Analyse email.", "error");
      return;
    }
    button.disabled = true;
    button.textContent = "Analysing...";
    showBanner(banner, "Analysing. The explanation runs LIME, so this takes a few seconds.", "info");
    try {
      const result = await apiCall("/explain", { method: "POST", body: { text: input.value } });
      clearChildren(banner);
      renderVerdict(result);
      renderTokens(document.getElementById("tokens"), result.text, result.tokens);
      renderSignals(result.tokens);
      const method = document.getElementById("method");
      clearChildren(method);
      method.appendChild(explainer("How this explanation was made", [
        `Method: ${result.method}. Model: ${result.model_id}.`,
        result.caveat,
        "A positive weight pushed the score towards phishing; a negative weight pushed it towards legitimate.",
      ]));
      showResults(true);
    } catch (err) {
      showResults(false);
      showBanner(banner, err.message === "Failed to fetch"
        ? "Cannot reach the PhishLens API. Start it with python run_api.py and reload this page."
        : err.message, "error");
    } finally {
      button.disabled = false;
      button.textContent = "Analyse email";
    }
  }

  input.addEventListener("input", updateCount);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) analyse();
  });
  button.addEventListener("click", analyse);
  document.getElementById("example").addEventListener("click", () => {
    input.value = EXAMPLE_EMAIL;
    updateCount();
  });
  document.getElementById("clear").addEventListener("click", () => {
    input.value = "";
    updateCount();
    clearChildren(banner);
    showResults(false);
    input.focus();
  });
  watchHealth(document.getElementById("pulse"));
});
