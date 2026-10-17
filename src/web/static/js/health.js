// health.js - keep the API status badge current by polling /health.
// Author: Colile Sibanda

const HEALTH_INTERVAL_MS = 8000;

/**
 * Poll GET /health and show the result in a .pulse badge.
 * Inputs: element - the .pulse span; onChange - optional callback given
 * true when the model is loaded and ready, false otherwise.
 * Output: none. Polling pauses while the tab is hidden.
 */
function watchHealth(element, onChange = () => {}) {
  async function tick() {
    let ready = false;
    if (!document.hidden) {
      try {
        const body = await apiCall("/health");
        ready = body.model_loaded === true;
        element.className = ready ? "pulse live" : "pulse idle";
        element.textContent = ready ? "API online, model loaded" : "API up, model still loading";
      } catch (err) {
        element.className = "pulse idle";
        element.textContent = "API offline: start it with python run_api.py";
      }
      onChange(ready);
    }
    setTimeout(tick, HEALTH_INTERVAL_MS);
  }
  tick();
}
