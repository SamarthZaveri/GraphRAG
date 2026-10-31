// Deployment hook, loaded before the app. Leave empty to use the page's own origin (the FastAPI
// app serves this file) or, on a static host with no /api, the recorded replay in data/replay.json.
// To point a static deployment at a live backend, set its HTTPS URL, e.g.:
//   window.LEDGER_API_BASE = 'https://your-space.hf.space';
window.LEDGER_API_BASE = window.LEDGER_API_BASE || '';
