// model.js - Model page: show the served model's provenance from
// /metadata and the stated limitations. Author: Colile Sibanda

// Plain names for the /metadata fields, in display order.
const METADATA_LABELS = [
  ["model_id", "Model id"],
  ["architecture", "Architecture"],
  ["threshold", "Decision threshold"],
  ["seed", "Random seed"],
  ["training_commit", "Training commit"],
  ["vocabulary_hash", "Vocabulary hash"],
  ["dataset_version", "Dataset version"],
];

/** Fill the detail list from a /metadata body. */
function renderMetadata(list, body) {
  clearChildren(list);
  for (const [key, label] of METADATA_LABELS) {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    const value = body[key];
    dd.textContent = typeof value === "number" && !Number.isInteger(value) ? fmt(value, 4) : String(value ?? EMPTY_CELL);
    list.append(dt, dd);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const banner = document.getElementById("page-banner");
  const list = document.getElementById("metadata");
  let loaded = false;

  async function load() {
    try {
      renderMetadata(list, await apiCall("/metadata"));
      clearChildren(banner);
      loaded = true;
    } catch (err) {
      showBanner(banner, err.message === "Failed to fetch"
        ? "Cannot reach the PhishLens API. Start it with python run_api.py; this page retries on its own."
        : err.message, "error");
    }
  }

  document.getElementById("limits").appendChild(explainer("Known limitations, in detail", [
    "Corpus markers: on pooled collections like these, a simple model can tell which collection an email came from with high accuracy. A phishing classifier can score well by learning those markers instead of phishing itself. The report measures this with a corpus probe and states it as a limitation.",
    "Generalisation: the headline scores come from a random split of the pooled data. Scores when one whole collection is held out (leave-one-corpus-out) are lower for most collections, and that gap is the honest estimate of performance on mail from a new source.",
    "Scope: one label per email, phishing or spam versus legitimate. Attachments, images and links are not inspected.",
  ]));

  watchHealth(document.getElementById("pulse"), (ready) => {
    if (ready && !loaded) load();
  });
});
