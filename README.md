# Ledger: Three Retrieval Engines for Financial Reports

Ledger answers financial-report questions and compares three retrieval engines on the same evidence-checked benchmark:

| Engine | Retrieval |
| --- | --- |
| `graphrag` | Graph facts and graph-selected source chunks for local search; community summaries for global search. No vector-chunk fallback. |
| `vector_rag` | Dense similarity search over raw document chunks in Chroma. |
| `hybrid_rag` | BM25 keyword retrieval plus dense vector retrieval, combined with reciprocal-rank fusion (RRF). No knowledge graph. |

All three use the same answer model and answer instructions so retrieval is the main difference. GraphRAG still uses text embeddings for entity linking and community ranking, plus optional R-GCN node embeddings for neighborhood expansion. Pure GraphRAG here means no chunks retrieved from the vector database.

## Architecture

- `backend/app/`: FastAPI, ingestion, graph and vector stores, the three engines, shared LLM transport, judging, and routing.
- `frontend/index.html`: served at `/` by the backend; supports all three engines and their comparisons.
- `RL/`: corpus collection, balanced benchmark generation, experiments, and offline router training.
- `scripts/`: model profile selection and provider checks.

Documents are chunked into both the graph and vector indexes. Deterministic table parsing and LLM extraction produce graph facts; entity resolution merges aliases. Leiden communities are summarized for global search. Hybrid search reuses the vector index's raw chunks and computes BM25 rankings over the current collection, so no separate ingestion step is needed. The current BM25 implementation scans the collection per request; a large deployment should use a persistent lexical index.

## Hosted model setup

Hosted APIs are the default. No local large-model weights or GPU are required. Chroma embeddings and optional R-GCN training still run in the backend.

The `14b`, `32b`, and `72b` profiles map to `Qwen/Qwen2.5-14B-Instruct`, `Qwen/Qwen2.5-32B-Instruct`, and `Qwen/Qwen2.5-72B-Instruct`. Start with 32B answers, 14B extraction, and 72B question/reference generation and judging. Larger models are candidates for improved quality, not guarantees; use measured results and review references.

These models currently have live Inference Provider mappings on Hugging Face. Provider availability, account access, pricing and supported parameters may change; `scripts/check_models.py` checks current mappings. Hugging Face routes large-model requests to hosted providers, rather than loading models on your PC. A dedicated compatible endpoint can be selected with `LEDGER_API_BASE_URL`.

Official resources: [HF chat-completion API](https://huggingface.co/docs/inference-providers/tasks/chat-completion), [14B model](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct), [32B model](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct), [72B model](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct).

### PowerShell quick start

Run from the project root. Use a Hugging Face token with Inference Providers permission and an account with available inference credits. Set the token in your environment; do not put it in source files.

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r backend/requirements-api.txt
python -m pip install -r RL/requirements.txt
$env:HF_TOKEN = Read-Host "Hugging Face token" -MaskInput
. .\scripts\use-model.ps1 -Profile 32b -Backend api
$env:LEDGER_EXTRACTION_MODEL = "Qwen/Qwen2.5-14B-Instruct"
$env:LEDGER_QUESTION_MODEL = "Qwen/Qwen2.5-72B-Instruct"
$env:LEDGER_JUDGE_MODEL = "Qwen/Qwen2.5-72B-Instruct"
python scripts/check_models.py
```

`-MaskInput` requires PowerShell 7; on Windows PowerShell 5.1 use a secure prompt:

```powershell
$ledgerCredential = Get-Credential -UserName "hf-token" -Message "Paste the HF token in the password field"
$env:HF_TOKEN = $ledgerCredential.GetNetworkCredential().Password
```

To verify credentials with a small completion from each configured model (uses credits):

```powershell
python scripts/check_models.py --smoke-test
```

Generate a balanced app benchmark from bundled documents, then ingest and evaluate:

```powershell
python RL/generate_benchmark.py --corpus backend/data/sample_docs --output backend/data/benchmark_questions.json --per-category 3
python backend/run_pipeline.py
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Open `http://localhost:8000`. The standalone pipeline replaces the active app indexes. You can instead start the app, ingest documents through its UI, and use the compare endpoint without running a benchmark.

## Balanced questions and evaluation

Generation requests each category independently, targeting three questions each: local, global, multi-hop, and conflict. It samples excerpts throughout each filing, requires verbatim supporting quotations, checks those quotes against original text, audits reference answers with the judge model, deduplicates questions, and retries deficits. If a balanced set cannot be supported, generation fails explicitly. `--allow-unbalanced` permits a documented shortfall; facts and conflicts are never fabricated to meet a quota.

Local questions require one specific fact; global questions synthesize across sources; multi-hop questions require distinct facts and reasoning; conflict questions compare like-for-like disclosures, including supported consistency. A changed quarterly value alone is not a changed definition or contradiction.

The judge sees shuffled anonymous answer labels, the reference, and its source evidence. Invalid or truncated responses are retried; failed evaluations are reported and excluded from rewards rather than converted into zero grades. Answers, references, quotations, rationales, failures, model IDs and retrieval version are saved for inspection.

Generated references and LLM audits can still be wrong. Review a sample manually. Training metrics are shuffled online simulations, not held-out production accuracy. Defaults use the same model family for generation and judging, so correlated errors remain possible.

## Experiments and training

```powershell
python RL/fetch_corpora.py --only semiconductors_2025
python RL/run_experiments.py --corpus semiconductors_2025 --output-dir RL/data/run-32b
python RL/train_query_bandit.py --results RL/data/run-32b/query_results.jsonl --seeds 20
python backend/ingest.py --documents RL/data/corpora/semiconductors_2025
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Check the SEC User-Agent in the fetcher first. Omit `--corpus` to evaluate all downloaded corpora. The ingest command replaces app indexes with the selected corpus; use `--mode add` to merge documents instead. The experiment runner uses isolated storage under its output directory by default; it does not replace app state. Saved benchmarks are shared under `RL/data/balanced_benchmarks`, so all answer-model comparisons reuse the same questions. Each corpus result is replaced on rerun, and aggregate JSONL files are rebuilt to avoid duplicated rows.

The eight-feature LinUCB policy chooses among all three engines using corpus structure and the question category. Live classification is LLM-based. GraphRAG maps `local` to neighborhood search and the other three categories to community-summary search; there are no dedicated multi-hop or conflict retrieval algorithms yet. The trained policy uses exploitation at serving time without live learning. Old two-engine or incompatible model/retrieval policies fall back to corpus-level rules.

Existing benchmark results and old reward files are historical. Fresh three-engine results use separate files and include retrieval-version metadata; old rewards are rejected by the query-policy trainer.

See [RL/README.md](RL/README.md) for a three-size comparison, index reuse, and training details.

## Deployment

The Docker image runs the app and frontend with hosted LLM calls; it does not load large-model weights. Configure secrets and role overrides in a local `.env` based on `.env.example`:

```powershell
Copy-Item .env.example .env
# Edit .env with HF_TOKEN and the desired model settings.
docker compose up --build -d
```

Open `http://localhost:8000`, then ingest documents. App state persists in the `ledger-data` volume. To use host-generated benchmark questions in the container:

```powershell
docker compose cp backend/data/benchmark_questions.json ledger:/data/benchmark_questions.json
```

The image omits PyTorch; R-GCN is optional and skipped. Hosted calls send supplied document context to the selected provider. The app remains a research tool; this deployment recipe serves it locally and does not add user authentication.

## Configuration and checks

| Variable | Purpose / default |
| --- | --- |
| `LEDGER_LLM_BACKEND` | `api` (or `ollama`) |
| `LEDGER_MODEL_PROFILE` | `14b`; selector script defaults to `32b` |
| `LEDGER_API_BASE_URL` | `https://router.huggingface.co/v1` |
| `HF_TOKEN` / `LEDGER_API_KEY` | API credential; `LEDGER_API_KEY` takes precedence |
| `LEDGER_EXTRACTION_MODEL` | Extraction and classification model override |
| `LEDGER_ANSWER_MODEL` | Shared answer model override |
| `LEDGER_QUESTION_MODEL` | Question/reference generation model override |
| `LEDGER_JUDGE_MODEL` | Reference audit and answer judge model override |
| `LEDGER_API_JSON_MODE` | `false`; enable only for providers supporting JSON response format |
| `LEDGER_DATA_DIR` | App state directory; defaults to `backend/data` |
| `LEDGER_OLLAMA_TIMEOUT` | Request timeout for either backend, `900` seconds |
| `LEDGER_NUM_CTX` | Ollama context window, `32768` |
| `LEDGER_QA_PER_CATEGORY` | `3` |
| `LEDGER_QA_INPUT_CHARS` | Source excerpt budget, `48000` characters |

```powershell
python -m pytest RL/tests -q
```

Offline tests cover BM25/RRF, graph-only context, API transport, balanced generation, judge failures, three-engine serving, and bandit learning. They do not measure real hosted-model answer quality.
