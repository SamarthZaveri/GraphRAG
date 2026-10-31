#!/bin/sh
# Start Ollama, seed the demo corpus on first boot, then serve the app on the Space port.
set -e
ollama serve > /tmp/ollama.log 2>&1 &
for i in $(seq 1 90); do
  curl -sf http://127.0.0.1:11434/api/tags > /dev/null && break
  sleep 1
done

if [ ! -f "$LEDGER_DATA_DIR/graph_state/graph.gpickle" ]; then
  if [ -n "$SEED_URL" ]; then
    echo "Seeding demo corpus from $SEED_URL"
    mkdir -p "$LEDGER_DATA_DIR"
    curl -fsSL "$SEED_URL" -o /tmp/seed.zip
    unzip -q -o /tmp/seed.zip -d "$LEDGER_DATA_DIR"
    rm -f /tmp/seed.zip
  else
    echo "SEED_URL not set: starting with an empty corpus (uploads still work, slowly, on CPU)."
  fi
fi

# Load the model now so the first visitor does not also wait for it to load.
curl -sf http://127.0.0.1:11434/api/generate -d '{"model":"qwen2.5:7b-instruct","prompt":"","keep_alive":"24h"}' > /dev/null || true

exec python -m uvicorn app.main:app --app-dir /app/backend --host 0.0.0.0 --port 7860
