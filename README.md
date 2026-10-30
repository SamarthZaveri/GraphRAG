# Ledger: adaptive retrieval over financial reports

Three retrieval engines answer questions over financial filings, and a learned router (LinUCB contextual bandit) picks one per question:

- **GraphRAG**: an entity graph built from the filings. Tables are parsed deterministically into one fact per cell; prose goes through LLM extraction. Local search ranks the facts and source chunks in the question's entity neighborhood; global search ranks community summaries and grounds each with its member chunks.
- **VectorRAG**: dense chunk retrieval (all-MiniLM-L6-v2 in Chroma).
- **HybridRAG**: BM25 and dense retrieval fused with reciprocal rank fusion.

Two benchmarks share the pipeline, with disjoint train/dev/test splits:

- `shared_finqa`: FinQA report pages (numeric questions scored exactly, plus generated text questions). Single-page lookups over tables.
- `edgar_xdoc`: real SEC earnings releases (8-K Item 2.02, Exhibit 99.1). Each corpus is a sector peer group, 3 companies x 2 quarters, so questions genuinely span documents. Splits are disjoint by sector.

Only source text enters the indexes; gold answers, programs and generated references stay outside retrieval.

## Run the experiment locally

Start Ollama and pull both models. For parallel requests, set `OLLAMA_NUM_PARALLEL=4` and `OLLAMA_FLASH_ATTENTION=1` in Ollama's environment and restart it.

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r RL/requirements.txt
ollama pull qwen2.5:7b-instruct
ollama pull qwen2.5:14b
$env:LEDGER_SEC_USER_AGENT = "Your Name you@example.com"   # SEC fair-access policy
.\scripts\run-local-flow.ps1                               # finqa -> edgar -> train -> router -> dev
.\scripts\run-local-flow.ps1 -StartAt test -Through test   # once, on the final system
```

Model profiles (`-Profile`, set by `scripts/use-ollama.ps1`):

| Profile | Extraction, classification, answers | Question generation, judge |
|---|---|---|
| `mixed` (default) | qwen2.5:7b-instruct | qwen2.5:14b |
| `7b` | qwen2.5:7b-instruct | qwen2.5:7b-instruct |
| `14b` | qwen2.5:14b | qwen2.5:14b |

On an 8 GB GPU the 7B runs fully on the GPU (about 40 tokens/s, about 94 tokens/s with 4 parallel requests), while the 14B runs half on the CPU (about 7 tokens/s). `mixed` puts the high-volume work on the 7B, and keeps the 14B where quality matters most: writing and auditing questions, and judging. The judge is then also a different, stronger model than the one answering.

Each stage checkpoints its work. Resume with `-StartAt finqa|edgar|train|router|dev|test`. Within a bundle, the benchmark ingests, then answers every question with all three engines in parallel, then scores. Answers and scores are checkpointed separately, so changing the judge or the scorer re-scores saved answers without re-answering them.

Results are written to `RL/data/runs/<profile>/`:
- per dataset: `<dataset>/<split>/{results,summary,analysis}.json`
- combined, which the UI reads: `<split>/`
- the policy: `policy.json`

`-LatencyWeight 0.2` trains a cost-aware router: reward = quality − 0.2 × min(latency / 60 s, 1).

Experiment tracking: every evaluation, router training and analysis is logged to MLflow, with each role's model, retrieval version, metric, dataset fingerprints, rewards, latencies, ingestion cost and output files. View it with `mlflow ui --backend-store-uri sqlite:///RL/mlflow.db`. Set `LEDGER_MLFLOW=0` to turn it off.

Numeric scoring compares an explicit final scalar with FinQA's gold execution value. It handles percent/ratio forms, the filing's scale and descriptive unit words, within rounding tolerance. It is **custom scalar-answer accuracy, not official FinQA program execution accuracy**. Text answers are scored by a blinded judge against audited references and evidence. Generated references and judge scores still need spot checks.

Router training gives each dataset equal weight. Within FinQA, numeric questions get 50% and each text category 12.5%. Only train rewards update the bandit, and held-out evaluation never does. The trained artifact is a small JSON policy; LLM weights are not fine-tuned.

## Serve the trained router

Use the model profile that produced the policy:

```powershell
. .\scripts\use-ollama.ps1 -Profile mixed
Copy-Item RL/data/runs/mixed/policy.json backend/data/graph_state/query_bandit.json
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Open http://localhost:8000, ingest documents and use automatic routing. The app rejects a policy trained with a different retrieval version or models.

For the existing Docker app with Ollama running on your host:

```powershell
Copy-Item .env.example .env
docker compose up --build -d
docker compose cp RL/data/runs/mixed/policy.json ledger:/data/graph_state/query_bandit.json
```

Ollama must be reachable from the container; configure its host binding appropriately. The app container stores its indexes in the persistent volume and does not contain model weights.

Optional R-GCN training requires `python -m pip install -r backend/requirements-rgcn.txt`. The default app and experiments use the base dependencies.

## Code layout

- `backend/app/`: ingestion, graph construction, retrieval, scoring and app routing. `table_parser.py` and `html_text.py` turn filing tables into deterministic facts.
- `RL/prepare_finqa.py`, `prepare_shared_benchmark.py`: FinQA bundles with shared numeric and text questions.
- `RL/prepare_edgar.py`: the cross-document SEC earnings-release benchmark.
- `RL/run_public_benchmark.py`: three-engine scoring per dataset and split.
- `RL/train_shared_router.py`: multi-dataset, optionally cost-aware router training.
- `RL/analyze_results.py`: bootstrap confidence intervals, routing behaviour, latency, per-dataset breakdown.
- `RL/tracking.py`: MLflow logging. `RL/rescore_numeric.py`: offline re-scoring after a scorer change.
- `scripts/run-local-flow.ps1`: the end-to-end workflow; `scripts/use-ollama.ps1`: model profiles.
- `.env.example`: Docker configuration; `backend/requirements-rgcn.txt`: optional graph training dependencies.

## Verification and details

```powershell
python -m pytest RL/tests -q
```

See [RL/README.md](RL/README.md) for individual stages, scoring rules and artifacts.
