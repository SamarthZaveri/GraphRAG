# Ledger: adaptive retrieval over financial reports

[![CI](https://github.com/SamarthZaveri/GraphRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/SamarthZaveri/GraphRAG/actions/workflows/ci.yml)

**Live replay:** _Vercel URL after deploy_ · **Live app:** _Hugging Face Space URL_ · **Demo video:** _link_ · **Deploy guide:** [docs/DEPLOY.md](docs/DEPLOY.md)

**The question:** when does a knowledge graph actually beat plain retrieval over financial filings, and can a learned router exploit the difference? The answer, on held-out dev and test splits: hybrid lexical + dense retrieval is the strongest single engine on both benchmarks; GraphRAG's edge on cross-document questions in training did not survive to the test sectors; and the learned router matches hybrid (significantly better than vector and GraphRAG) but does not beat it, because the best engine changes question to question in ways the router's features cannot predict.

Three retrieval engines answer questions over financial filings, and a learned router (LinUCB contextual bandit) picks one per question:

- **GraphRAG**: an entity graph built from the filings. Tables are parsed deterministically into one fact per cell; prose goes through LLM extraction. Local search ranks the facts and source chunks in the question's entity neighborhood; global search ranks community summaries and grounds each with its member chunks.
- **VectorRAG**: dense chunk retrieval (all-MiniLM-L6-v2 in Chroma).
- **HybridRAG**: BM25 and dense retrieval fused with reciprocal rank fusion.

Two benchmarks share the pipeline, with disjoint train/dev/test splits:

- `shared_finqa`: FinQA report pages (numeric questions scored exactly, plus generated text questions). Single-page lookups over tables.
- `edgar_xdoc`: real SEC earnings releases (8-K Item 2.02, Exhibit 99.1). Each corpus is a sector peer group, 3 companies x 2 quarters, so questions genuinely span documents. Splits are disjoint by sector.

Only source text enters the indexes; gold answers, programs and generated references stay outside retrieval.

## Results

Mean reward (0–1) with bootstrap 95% CIs. The router is trained on train only; dev and test sectors never appear in training.

| Split | Questions | GraphRAG | Vector | Hybrid | Router | Oracle |
|---|---|---|---|---|---|---|
| train (in-sample) | 164 | 0.54 | 0.63 | **0.71** | n/a | 0.85 |
| dev | 88 | 0.60 | 0.68 | 0.79 | **0.79** | 0.85 |
| test | 88 | 0.44 | 0.60 | 0.67 | **0.67** | 0.78 |

On test the router is significantly better than vector (+0.077, 95% CI [0.011, 0.148]) and GraphRAG (+0.230), and level with hybrid (+0.002, CI [−0.014, 0.018]). It picks hybrid for 85% of test questions and vector for the rest, never GraphRAG; it picks the best available engine 77% of the time.

By dataset (test): on single-page **FinQA**, hybrid 0.68, vector 0.65, GraphRAG 0.45; the router 0.68. On cross-document **SEC EDGAR**, hybrid 0.66, vector 0.49, GraphRAG 0.43; the router 0.67. In training GraphRAG beat vector on EDGAR (0.60 vs 0.54) and on dev the two tied (0.63 vs 0.62), but on the test sectors it lost. The oracle gap (0.78 vs 0.67 on test) is per-question: no engine wins a whole category, which bounds what a category-level router can learn.

Retrieval quality follows the same order. Document-level nDCG@5 on test: hybrid 0.95, vector 0.89, GraphRAG 0.83; on EDGAR, GraphRAG (0.81) ranks the right filings above vector (0.77) but below hybrid (0.87), so its answer-quality loss there comes after retrieval.

Full numbers: `releases/<tag>/` (frozen results, policy, manifest with commit, models and dataset fingerprints) and the replay site.

### Engineering notes (what broke and how it was fixed)

- **Scorer bug inflated failures**: a bare `12.5` against a gold `12.5%` scored zero. Numeric scorer v2 handles percent/ratio forms, the filing's scale and unit words; old answers were re-scored offline from checkpoints (`RL/rescore_numeric.py`).
- **Tables lost in extraction**: the LLM dropped table rows. Tables are now parsed deterministically (FinQA rows, EDGAR HTML with colspans and split `$`/`)` cells) into one fact per cell.
- **GraphRAG silently capped at 800 facts** in discovery order, dropping ~70% of a large neighborhood unranked. Embeddings are now cached and the whole neighborhood is ranked: retrieval went from 16 s to 0.12 s per query.
- **Benchmark questions with unlabeled numbers**: the generator cut table rows mid-line, so multi-hop questions failed audit. Passages now break only between lines.
- **Router training depended on question order**: replaying train questions as an online bandit (updating only the picked engine) gave dev scores from 0.69 to 0.80 depending on the shuffle. Offline, every engine's reward is known, so training now uses all of them (`--feedback full`): deterministic, and it matches hybrid on dev (0.791 vs 0.789).
- **Small-model failure modes**: extraction loops on a few chunks are split instead of retried with bigger budgets; runaway answers get one bigger budget, then score zero instead of aborting the run.

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

## Serving and MLOps

- **Streaming:** `POST /api/query/stream` sends server-sent events: the routing decision, then answer text as it is generated, then the full response. Time to first token is in each response's metrics and in `/api/metrics` (p50/p95).
- **Response cache:** a repeated question (normalized) with the same engine is served from SQLite. The cache key includes content hashes of the corpus, the router policy, the models and the retrieval version, so re-ingesting invalidates it. `/api/metrics` reports the hit rate separately from model latency.
- **Upload limits:** 5 files × 10 MB, PDFs up to 60 pages (`LEDGER_MAX_PDF_PAGES`), with encrypted and scanned PDFs rejected.
- **Retrieval metrics:** document-level hit rate, MRR, precision@k, recall@k and nDCG@k (k = 1, 3, 5) per engine and dataset, in each split's `analysis.json` and in MLflow.
- **Model registry:** each trained policy is registered as a version of the MLflow model `ledger-router` (a pyfunc: context in, engine and predicted rewards out, `RL/router_pyfunc.py`). `freeze_release.py` points the `champion` alias at the released policy and tags it with its held-out scores.
- **CI quality gate:** `scripts/quality_gate.py` checks the newest release. It fails the build if the files no longer match the manifest, a run is incomplete, or the router falls more than 0.02 below the best single engine or is significantly worse than any engine.

## Code layout

- `backend/app/`: ingestion, graph construction, retrieval, scoring and app routing. `table_parser.py` and `html_text.py` turn filing tables into deterministic facts.
- `RL/prepare_finqa.py`, `prepare_shared_benchmark.py`: FinQA bundles with shared numeric and text questions.
- `RL/prepare_edgar.py`: the cross-document SEC earnings-release benchmark.
- `RL/run_public_benchmark.py`: three-engine scoring per dataset and split.
- `RL/train_shared_router.py`: multi-dataset, optionally cost-aware router training.
- `RL/analyze_results.py`: bootstrap confidence intervals, routing behaviour, latency, per-dataset breakdown.
- `RL/tracking.py`: MLflow logging. `RL/rescore_numeric.py`: offline re-scoring after a scorer change.
- `scripts/run-local-flow.ps1`: the end-to-end workflow; `scripts/use-ollama.ps1`: model profiles.
- `backend/app/response_cache.py`: answer cache. `RL/router_pyfunc.py`: MLflow model for the policy. `scripts/quality_gate.py`: CI check on frozen results. `scripts/warm_cache.py`: pre-answers questions for slow deployments.
- `scripts/export_replay.py`: static replay data for the frontend; `scripts/freeze_release.py`: versioned results releases; `scripts/build_demo_corpus.ps1`: pre-ingested corpus for public deployments.
- `deploy/hf-space/`: free Hugging Face Space (Ollama 7B + app); `vercel.json`: static replay site; `.github/workflows/ci.yml`: tests, lint, Docker build.
- `.env.example`: Docker configuration; `backend/requirements-rgcn.txt`: optional graph training dependencies.

## Verification and details

```powershell
python -m pytest RL/tests -q
```

See [RL/README.md](RL/README.md) for individual stages, scoring rules and artifacts.
