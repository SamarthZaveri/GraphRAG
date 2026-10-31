# Deploying Ledger

Everything here is free except the optional AWS GPU, which runs on credits and is started only for demos.

| Target | What visitors get | Cost | Speed |
|---|---|---|---|
| **1. Vercel replay** (primary public link) | Benchmarks, recorded answers from all three engines with judge verdicts, the router's decisions | $0 | Instant, never sleeps |
| **2. Hugging Face Space** (live demo) | The full app on a pre-ingested EDGAR corpus | $0 | ~1 min per answer on 2 free vCPUs; sleeps when idle |
| **3. Your PC + Cloudflare Tunnel** (interviews) | The full app on your GPU | $0 | Fast; up only while your PC is on |
| **4. AWS g4dn.xlarge** (optional) | The full app on a T4 GPU | ~$0.53/hr, from credits | Fast; start and stop it per demo |

The frontend picks its mode automatically: with a reachable backend it is the live app; on a static host
with no `/api` it loads `frontend/data/replay.json` and shows a "replay" ribbon. `?replay` in the URL
forces replay mode anywhere.

---

## 0. After the test split: freeze and export

```powershell
python RL/analyze_results.py --run-dir RL/data/runs/mixed --split test   # if not already run by the flow
python scripts/export_replay.py                     # -> frontend/data/replay.json (showcase from test)
git add -A; git commit -m "Final results"           # freeze needs a clean tree
python scripts/freeze_release.py --tag v1-router    # -> releases/v1-router/ with MANIFEST.json
git add releases/v1-router; git commit -m "Release v1-router"; git tag v1-router; git push --tags origin main
```

## 1. Vercel replay

1. vercel.com → Add New Project → import `SamarthZaveri/GraphRAG`.
2. Framework preset: **Other**. Leave the build command empty. `vercel.json` already sets
   `outputDirectory: frontend`.
3. Deploy. Every push to `main` redeploys; re-run `export_replay.py` after new results.

## 2. Hugging Face Space (live, CPU)

```powershell
.\scripts\build_demo_corpus.ps1                       # -> dist/demo_seed.zip (graphs, indexes, policy, cached answers)
gh release create v1-router dist/demo_seed.zip --notes "Demo corpus and router policy"
```

1. huggingface.co → New Space → SDK **Docker** → blank, CPU basic (free).
2. Add `deploy/hf-space/Dockerfile`, `start.sh` and `README.md` to the Space repo (web upload or git).
3. Space settings → Variables: `SEED_URL` = the release asset URL
   (`https://github.com/SamarthZaveri/GraphRAG/releases/download/v1-router/demo_seed.zip`),
   `LEDGER_CORS_ORIGINS` = your Vercel URL.
4. The first build takes ~15 minutes (it bakes in the 7B). The Space sleeps after inactivity; the first
   visit after that waits for it to wake up.

**Running on CPU.** The free tier has 2 vCPUs, so a 7B model writes a few tokens a second and a new
answer takes 1–3 minutes. The image sets `LEDGER_INFERENCE_DEVICE=cpu`, which makes the site show a
CPU notice. Three things keep it usable:
- **Streaming:** answers come from `/api/query/stream` (server-sent events) and appear word by word.
  The time to first token is recorded in the telemetry.
- **Pre-answered questions:** `build_demo_corpus.ps1` runs `scripts/warm_cache.py` on your GPU to
  answer 6 questions per corpus (Auto and Compare). It ships them in the seed's
  `response_cache.sqlite3`, and the site offers them as "instant" chips. Use `-WarmPerCorpus 0` to
  skip this step, which takes about a minute per question.
- **Response cache:** any repeated question on an unchanged corpus is answered from the cache.
  Re-ingesting changes the corpus hash, so a stale answer is never served.

Per-visitor rate limits are tighter here (12 questions and 4 comparisons per 10 minutes), and only one
LLM request runs at a time.

The image clones this repo at build time, so the GitHub repo must be public. The git history was
scanned for keys before publishing; `secrets.txt` and `.env` are gitignored.

**Vercel frontend + Space backend.** To make the Vercel site live, set
`window.LEDGER_API_BASE = 'https://<user>-<space>.hf.space'` in `frontend/config.js`. Leave it empty to
keep Vercel as the instant replay and link to the Space separately, which is better for a demo: replay
never sleeps, so it never fails.

## 3. Your PC + Cloudflare Tunnel (for interviews)

```powershell
. .\scripts\use-ollama.ps1 -Profile mixed
Copy-Item RL/data/runs/mixed/policy.json backend/data/graph_state/query_bandit.json
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
winget install Cloudflare.cloudflared
cloudflared tunnel --url http://localhost:8000      # prints a temporary https://*.trycloudflare.com URL
```

The tunnel URL is HTTPS, so it works from any browser with no mixed-content issues. Set
`LEDGER_PUBLIC_MODE=true` first if strangers will use it.

## 4. AWS g4dn.xlarge on credits (optional)

Before anything else: **Billing → Budgets → create a $10 monthly cost budget with an email alert.**

1. Service Quotas → EC2 → "Running On-Demand G and VT instances" → request 4 vCPUs. New accounts often
   start at 0, and approval can take a day.
2. Launch a **Deep Learning Base GPU AMI (Ubuntu)** on `g4dn.xlarge`, 60 GB gp3. Security group: 22
   from your IP only. No public 8000; the tunnel or Caddy handles HTTPS.
3. On the instance:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh && ollama pull qwen2.5:7b-instruct
   git clone https://github.com/SamarthZaveri/GraphRAG && cd GraphRAG
   python3 -m venv venv && . venv/bin/activate && pip install -r backend/requirements.txt
   curl -fsSL "$SEED_URL" -o seed.zip && mkdir -p backend/data && unzip -o seed.zip -d backend/data
   LEDGER_PUBLIC_MODE=true LEDGER_EXTRACTION_MODEL=qwen2.5:7b-instruct LEDGER_QUERY_MODEL=qwen2.5:7b-instruct \
     LEDGER_ANSWER_MODEL=qwen2.5:7b-instruct LEDGER_JUDGE_MODEL=qwen2.5:14b LEDGER_QUESTION_MODEL=qwen2.5:14b \
     LEDGER_GPU_HOURLY_USD=0.526 nohup python -m uvicorn app.main:app --app-dir backend --port 8000 &
   cloudflared tunnel --url http://localhost:8000     # or Caddy with a domain for a stable HTTPS URL
   ```
4. **Stop the instance after each demo.** A stopped instance costs only its disk (~$5/month for 60 GB).

## HTTPS and mixed content

A Vercel page (HTTPS) cannot call an `http://` backend: the browser blocks it. Every live option above
is HTTPS (hf.space, Cloudflare Tunnel, or Caddy with a domain). Never point `LEDGER_API_BASE` at a bare
`http://IP:8000`.

## Serving with a hosted API instead of Ollama

`LEDGER_LLM_BACKEND=api` with `LEDGER_API_BASE_URL`, an API key in the environment, and the provider's
name for the same model also works. The router policy stays valid: `corpus_router.canonical_model`
compares model families across transports. Quantization differs from Ollama's Q4 build, so say
"same model, different serving stack" when quoting benchmark numbers next to it.
