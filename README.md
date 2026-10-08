# Ledger: GraphRAG and Vector RAG for Financial Reports

Ledger answers questions about financial reports using a knowledge graph or vector retrieval. It compares both engines on the same questions and uses a learned contextual bandit to select an engine for each incoming question.

## Project structure

| Path | Purpose |
| --- | --- |
| `backend/app/` | FastAPI application, ingestion, retrieval, answering, evaluation, and live routing |
| `frontend/index.html` | Static interface for asking questions, comparing engines, and exploring the graph |
| `RL/` | Corpus collection, experiments, ablation, and offline router training |
| `backend/data/` | Active document indexes, graph state, saved policy, and app benchmark results |
| `RL/data/` | Experiment corpora, recorded rewards, and ablation results |

RL scripts reuse the backend implementation. The backend does not import the RL folder; it loads the policy that training saves into its state directory.

## Ingestion and retrieval

Each document is indexed by both engines:

```text
Financial reports
  +-- Graph pipeline
  |     chunks -> deterministic table parsing + LLM entity/relation extraction
  |     -> entity resolution -> NetworkX knowledge graph
  |     -> community summaries + optional R-GCN node embeddings
  |
  +-- Vector pipeline
        chunks -> local embeddings -> persistent Chroma collection
```

Entity resolution combines string matching, semantic similarity, and LLM confirmation for uncertain semantic matches. Community detection tries several Leiden resolutions, with a NetworkX fallback. R-GCN embeddings are trained through link prediction and support graph neighborhood expansion; they do not classify questions or select an engine.

GraphRAG has two retrieval modes:

- **Local:** match query entities to graph nodes, expand two hops, add R-GCN neighbors, and retrieve facts and supporting source chunks. By default, it also includes up to four vector-retrieved chunks.
- **Global:** rank community summaries by text similarity and synthesize an answer from up to four summaries.

Vector RAG retrieves up to six raw document chunks from Chroma. Both engines use the configured answer model. The standard local GraphRAG comparison therefore measures hybrid graph-plus-vector retrieval against vector retrieval. `RL/run_ablation.py` disables the local vector fallback; global search is unchanged.

## Query classification and routing

There are two decisions: which engine answers, and which retrieval mode GraphRAG uses.

`query_engine.classify_query()` prompts the extraction LLM to return a category and up to four entity names:

| Category | Intended question type | GraphRAG automatic mode |
| --- | --- | --- |
| `local` | Specific facts about named companies, metrics, or periods | Local |
| `global` | Broad themes or synthesis across documents | Global |
| `multi_hop` | Connect two or more specific facts | Global |
| `conflict` | Check consistency or contradictions | Global |

Classification is prompt-based, rather than a separately trained classifier. Missing or invalid categories default to `global`. Multi-hop and conflict questions currently share global retrieval rather than having dedicated retrieval strategies.

For `/api/query` with `engine="auto"`, `corpus_router.route_query()` loads `backend/data/graph_state/query_bandit.json`. Its eight-dimensional context contains:

1. Fraction of entities appearing in two or more documents.
2. Maximum entity document recurrence divided by ten, capped at one.
3. Document count divided by ten, capped at one.
4. Four one-hot question-category features.
5. A bias term.

The router chooses the engine with the highest predicted reward, without exploration or updates during live requests. If the policy is unavailable or routing fails, it falls back to corpus-level rules using document count, shared entities, community modularity, and R-GCN validation AUC. Modularity and AUC are not inputs to the learned query-level policy.

Users can force `graphrag` or `vector_rag`; GraphRAG also accepts `auto`, `local`, or `global` mode. The compare endpoint and benchmark always run both engines, independently of the engine router.

## Running the app

Run these commands from the project root. Install Ollama separately and ensure the configured model is available.

```powershell
python -m pip install -r backend/requirements.txt
ollama pull qwen2.5:7b-instruct
ollama serve
```

If Ollama is already running, skip `ollama serve`. In another terminal:

```powershell
cd backend
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open `frontend/index.html` in a browser. Its default backend address is `http://localhost:8000`. Ingest the bundled sample documents or upload reports before querying.

The standalone pipeline ingests the bundled samples and runs the app benchmark:

```powershell
cd backend
python run_pipeline.py
```

This replaces the active graph and vector index.

### Configuration

| Environment variable | Default |
| --- | --- |
| `OLLAMA_HOST` | `http://localhost:11434` |
| `LEDGER_EXTRACTION_MODEL` | `qwen2.5:7b-instruct` |
| `LEDGER_ANSWER_MODEL` | `qwen2.5:7b-instruct` |
| `LEDGER_JUDGE_MODEL` | `qwen2.5:7b-instruct` |
| `LEDGER_OLLAMA_TIMEOUT` | `180` seconds |
| `LEDGER_EXTRACTION_CONCURRENCY` | `4` |

Inference uses local Ollama models. Initial model and embedding downloads, dependency installation, and SEC corpus collection require network access.

## Experiments and router training

See [RL/README.md](RL/README.md) for the full workflow.

```powershell
python -m pip install -r RL/requirements.txt
cd RL
python fetch_corpora.py
python run_experiments.py --corpus semiconductors_2025
python train_query_bandit.py
```

Check `SEC_USER_AGENT` in `fetch_corpora.py` before fetching. Omit `--corpus` to evaluate all available corpus folders.

**Experiment ingestion shares the backend's active storage.** Experiments and ablations reset the vector collection and overwrite graph state, summaries, and R-GCN artifacts. Re-ingest the desired app corpus afterward. Experiment results append to JSONL files, so repeated runs add training rows rather than replacing earlier results.

## Evaluation and limits

The benchmark runs both engines on the same questions and uses an LLM judge to score answers against references on a 0-5 rubric. `backend/data/benchmark_results.json` stores the app's latest benchmark; the query router trains from `RL/data/query_results.jsonl` instead.

Interpret results within the evaluated corpus and question set. A higher average in one saved run does not establish a general advantage. Generated references and judge scores can be wrong, and unparseable judge output currently becomes zero scores. Question-category coverage is uneven, and the training script reports shuffled online-simulation performance without a separate held-out evaluation.

Current implementation also repeats classification calls across routing and local retrieval. Reusing one classification, adding dedicated multi-hop/conflict retrieval, isolating experiment storage, and validating the router on held-out corpora are useful next steps.
