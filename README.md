# Ledger: adaptive retrieval over financial reports

The active workflow uses **local Ollama Qwen2.5 14B for every LLM role**:

`same FinQA report excerpts/tables -> numeric + text questions -> three RAG answers -> scores -> trained adaptive router -> held-out validation/test`

FinQA's original numeric questions and generated local/global/multi-hop/conflict questions share the exact same document bundles. Only source text/tables enter indexes. Gold answers, programs and generated QA remain outside retrieval. FinQA provides report-page excerpts; this workflow does not fetch complete original filings.

GraphRAG retrieves graph-linked source chunks and community summaries without vector-chunk fallback. VectorRAG uses dense chunk retrieval. Hybrid combines BM25 and dense retrieval through reciprocal rank fusion. All three share the same answer model. The learned LinUCB router selects an engine using document structure and question-only category features; serving uses the trained policy without online updates. GraphRAG currently maps multi-hop/conflict categories to community-summary retrieval rather than dedicated algorithms.

## Run locally

Start Ollama, activate the Python environment, and pull the models:

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r RL/requirements.txt
ollama pull qwen2.5:14b
. .\scripts\use-ollama.ps1
python scripts/check_models.py --smoke-test
```

Extraction, query classification, question generation, reference auditing, answering and text judging all use Qwen2.5 14B in separate calls. The helper sets every model role explicitly. Context is 16384 tokens, calls are sequential, and models unload after requests (`keep_alive=0`) to limit simultaneous memory use. Actual speed and memory depend on GPU VRAM and CPU offloading.

Prepare FinQA once (existing downloads/prepared files can be reused), then run the shared-source workflow:

```powershell
python RL/prepare_finqa.py --train-limit 40 --dev-limit 20 --test-limit 20
.\scripts\run-local-flow.ps1
```

The runner stops after validation by default. Choose settings on dev, then run the final test separately with `scripts/run-local-flow.ps1 -StartAt test -Through test`.

Results and the trained policy are stored under `RL/data/local-14b`. Each bundle gets 12 custom questions by default; `-PerCategory 5` targets 20 and uses separate dataset/run folders with a `-c5` suffix. The runner stops after failures and accepts `-StartAt prepare|train|router|dev|test` for recovery. Successful generation, extraction and scored questions are checkpointed.

Train/dev/test report separation is established before custom generation. Settings can be chosen on dev; freeze them before interpreting the test results. FinQA retains its numeric question distribution. Custom generation strictly requires the requested count for each category and fails if source evidence is insufficient. The default subsets are small smoke experiments, not reliable production estimates.

Numeric scoring directly compares an explicit final scalar with FinQA's gold execution value, handling percent/ratio and documented scale conversions plus rounding tolerance. It is **custom scalar-answer accuracy, not official FinQA program execution accuracy**. Text answer scores come from a blinded local 14B judge against audited references/evidence. Numeric/text results and text categories are reported separately. Generated references and judge scores need manual review; a 14B judge can still make correlated errors.

Training defaults to 50% numeric weight and 12.5% per text category. Only train rewards update the bandit; held-out evaluation never updates it. The trained artifact is a small JSON policy; LLM weights are not fine-tuned.

## Serve the trained router

Use the model profile that produced the policy:

```powershell
. .\scripts\use-ollama.ps1
Copy-Item RL/data/local-14b/policy.json backend/data/graph_state/query_bandit.json
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Open http://localhost:8000, ingest application documents, and select automatic routing. A policy does not copy benchmark indexes into app state. Use matching extraction/answer models; the app rejects incompatible policies.

For the existing Docker app with Ollama running on your host:

```powershell
Copy-Item .env.example .env
docker compose up --build -d
docker compose cp RL/data/local-14b/policy.json ledger:/data/graph_state/query_bandit.json
```

Ollama must be reachable from the container; configure its host binding appropriately. The app container stores its indexes in the persistent volume and does not contain model weights.

Optional R-GCN training requires `python -m pip install -r backend/requirements-rgcn.txt`. The default app and experiments use the base dependencies.

## Code layout

- `backend/app/`: ingestion, graph construction, retrieval, scoring and app routing.
- `RL/prepare_finqa.py` and `prepare_shared_benchmark.py`: source bundles and shared numeric/text questions.
- `RL/run_public_benchmark.py`: three-engine train/validation/test scoring.
- `RL/train_shared_router.py`: weighted training and policy export.
- `scripts/run-local-flow.ps1`: the local 14B workflow.
- `.env.example`: Docker configuration; `backend/requirements-rgcn.txt`: optional graph training dependencies.

## Verification and details

```powershell
python -m pytest RL/tests -q
```

See [RL/README.md](RL/README.md) for individual stages, scoring rules and artifacts.
