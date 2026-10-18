// about.js - About page: fill the live facts from /metadata once the model is loaded.
// Author: Colile Sibanda

document.addEventListener("DOMContentLoaded", () => {
  const banner = document.getElementById("page-banner");
  let loaded = false;

  /** Write one live fact into its <dd>. */
  function setFact(id, value) {
    document.getElementById(id).textContent = value;
  }

  async function load() {
    try {
      const meta = await apiCall("/metadata");
      setFact("fact-model", `${meta.model_id} (${meta.architecture})`);
      setFact("fact-threshold", `${fmt(meta.threshold, 4)}, chosen on the validation set`);
      setFact("fact-commit", meta.training_commit.slice(0, 7));
      clearChildren(banner);
      loaded = true;
    } catch (err) {
      showBanner(banner, err.message === "Failed to fetch"
        ? "Cannot reach the PhishLens API, so the live facts are unavailable. The rest of this page does not need it."
        : err.message, "info");
    }
  }

  watchHealth(document.getElementById("pulse"), (ready) => {
    if (ready && !loaded) load();
  });
});
