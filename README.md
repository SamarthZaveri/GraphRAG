# Ledger — GraphRAG for Contract & Financial Document Intelligence

A from-scratch GraphRAG system for contracts/financial filings, benchmarked
head-to-head against vanilla vector RAG on the same multi-hop question set —
**fully offline**, using a local [Ollama](https://ollama.com) model instead
of a hosted API.

```
Documents ──▶ Chunking ──▶ LLM Entity/Relation Extraction (Ollama)
                                     │
                                     ▼
                          Knowledge Graph (NetworkX)
                                     │
                        Community Detection (Leiden)
                                     │
                     Hierarchical Community Summaries (Ollama)
                                     │
Question ──▶ Router (local vs global) ─┴──▶ Graph Traversal / Summary Retrieval
                                                        │
                                                        ▼
                                               Grounded Answer + Citations
```

A vanilla vector-RAG baseline (Chroma + Ollama generation, no graph) is built
alongside it so the two can be compared on the same 12-question benchmark
(local / global / multi-hop / conflict-detection questions) with an
LLM-judge rubric.

## 0. Install Ollama and pull a model

```bash
# see https://ollama.com for your OS
ollama pull llama3.1      # or any other model you have pulled
ollama serve               # usually already running as a background service
```

Nothing else needs a network connection or API key. (One exception: Chroma's
default embedding function downloads a small ~80MB ONNX model the *first*
time you run ingestion — after that it's cached locally and fully offline too.)

If you want to use a different model, or split extraction/answering/judging
across different local models, set these before starting the backend:

```bash
export OLLAMA_HOST=http://localhost:11434     # default
export LEDGER_EXTRACTION_MODEL=llama3.1       # used for extraction, routing, community summaries
export LEDGER_ANSWER_MODEL=llama3.1           # used for final answer generation
export LEDGER_JUDGE_MODEL=llama3.1            # used for benchmark scoring
```

A bigger/smarter model for `LEDGER_ANSWER_MODEL` and `LEDGER_JUDGE_MODEL`
(e.g. `qwen2.5:14b`) and a faster small one for `LEDGER_EXTRACTION_MODEL`
(e.g. `llama3.2:3b`) is a good split if your machine can run more than one
model size.

## 1. Run the backend

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The API docs are then at http://localhost:8000/docs.

## 2. Open the frontend

The frontend is a single static file — no build step, no npm install.

```bash
open frontend/index.html
```

(or just double-click it / drag it into a browser tab). It talks to the
backend at `http://localhost:8000/api` by default. If you're running the
backend somewhere else, open the browser console and run:

```js
localStorage.setItem('ledger_api_base', 'http://your-host:8000')
```
then reload.

From the sidebar: click **"Ingest 3 sample SEC filings"** (real public SEC
EDGAR exhibits already bundled in `backend/data/sample_docs/` — a sponsor
support agreement + press release from one SPAC merger, plus an unrelated
financial-advisory engagement letter, chosen so there are genuine
cross-document entities *and* one unrelated document to test that the system
doesn't invent false connections). Then explore the **Knowledge graph**,
**Ask** questions (with a live GraphRAG vs. vector-RAG comparison), browse
**Communities**, and **Run benchmark**.

You can also upload your own `.txt`/`.pdf` contracts from the sidebar instead
of the bundled samples.

## 3. Or run everything from the command line

```bash
cd backend
python run_pipeline.py
```

This ingests the sample docs, builds the graph + communities, runs the
12-question benchmark, and prints a report — no server/frontend needed.
Useful for a fast sanity check or for grabbing numbers for a writeup.

## How the pieces fit together

- **`app/extraction.py`** — chunks documents and calls the local model with a
  strict JSON schema to pull out `(entity, relation, entity)` triples.
- **`app/graph_store.py`** — builds a NetworkX `MultiDiGraph` from the
  triples, with a lightweight string-similarity entity resolver (exact +
  fuzzy match) as an MVP stopgap for merging near-duplicate names — real
  resolution is a Phase 2 item per the PRD.
- **`app/community.py`** — runs Leiden community detection (via
  `igraph`/`leidenalg`, falling back to networkx's greedy-modularity if those
  aren't importable) and asks the local model to summarize each cluster.
- **`app/query_engine.py`** — the local/global router + local graph
  traversal + global community-summary retrieval. This local/global split is
  what makes it GraphRAG rather than vector RAG.
- **`app/vector_baseline.py`** — the vanilla comparison system: chunk, embed
  with Chroma's local embedding function, top-k similarity search, stuff into
  the prompt. No graph.
- **`app/benchmark.py`** — runs `data/benchmark_questions.json` against both
  systems and scores each answer 0–5 against a reference answer with an
  LLM-judge rubric.
- **`app/ollama_client.py`** — the only place that talks to Ollama's
  `/api/chat` endpoint; swap this out if you'd rather point at a different
  local runtime (llama.cpp server, vLLM, LM Studio, etc.) — everything else
  is provider-agnostic.

## Known limitations (honest, per the PRD's own risk table)

- Entity resolution is exact/fuzzy string matching only — it will not merge
  "Acme Corp" and "the Company" unless the extraction step already
  normalized them. Real coreference/alias resolution is Phase 2.
- Extraction quality depends entirely on which local model you point at —
  small/quantized models will produce noisier triples than a frontier
  hosted model would. Try a bigger model for `LEDGER_EXTRACTION_MODEL` if
  the graph looks sparse or wrong.
- The benchmark's LLM judge is only as reliable as the local model doing the
  judging — treat the 0–5 scores as directional, not authoritative, and read
  the `judge_rationale` column, not just the number.
