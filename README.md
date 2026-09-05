# Ledger — GraphRAG for Financial Reports

A from-scratch GraphRAG system tuned specifically for **financial reports**
(earnings releases, 10-K/10-Q style filings), benchmarked head-to-head
against vanilla vector RAG on the same multi-hop question set — fully
offline via a local [Ollama](https://ollama.com) model.

```
Documents ──▶ Chunking ──▶ LLM Entity/Relation Extraction (Ollama)
                                     │
                                     ▼
                    Knowledge Graph (NetworkX) ◀── Tiered entity resolution
                                     │              (string + embedding + LLM-confirmed)
                        Community Detection (Leiden)
                                     │              R-GCN node embeddings
                     Hierarchical Community Summaries    (self-supervised
                                     │                     link prediction)
Question ──▶ Router (local vs global) ─┴──▶ Graph traversal (2-hop + R-GCN  
                                             neighborhood) + raw source text
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
ollama pull qwen2.5:3b-instruct    # default model — good instruction-following for JSON
ollama serve                        # usually already running as a background service
```

Nothing else needs a network connection or API key. (One exception: Chroma's
default embedding function, and the entity-resolution/retrieval-ranking
embeddings, download a small ~80MB ONNX model the *first* time you run
ingestion — after that it's cached locally and fully offline too.)

Override the model per stage if you want:

```bash
export OLLAMA_HOST=http://localhost:11434
export LEDGER_EXTRACTION_MODEL=qwen2.5:3b-instruct   # extraction, routing, community summaries
export LEDGER_ANSWER_MODEL=qwen2.5:3b-instruct        # final answer generation
export LEDGER_JUDGE_MODEL=qwen2.5:3b-instruct         # benchmark scoring
```

## 1. Run the backend

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # now includes torch, for R-GCN embeddings
uvicorn app.main:app --reload --port 8000
```

R-GCN is optional — if `torch` isn't installed, everything else still works;
the sidebar will just show "R-GCN unavailable."

## 2. Open the frontend

Still a single static file — no build step, no npm install.

```bash
open frontend/index.html
```

Drag-and-drop `.txt`/`.pdf` files onto the sidebar dropzone, or click it to
browse. Choose **Add to corpus** (incremental — keeps what's already
ingested, only extracts genuinely new files) or **Replace corpus** (wipes
and rebuilds from just what you're uploading).

## 3. Or run everything from the command line

```bash
cd backend && python run_pipeline.py
```

---

## What changed in this round

### 1. Phase 2 (from the original PRD's stretch goals)

- **Real entity resolution**, not just fuzzy string matching. Three tiers in
  `graph_store.py`: (1) exact/fuzzy name match, (2) local sentence-embedding
  similarity between "name + type + description" for same-type entities,
  auto-merging above a high threshold, and (3) for borderline embedding
  matches, a one-shot LLM confirmation before merging — so a coincidentally
  similar description can't silently merge two different companies.
- **Incremental ingestion.** `POST /api/ingest/upload` takes a `mode` of
  `add` or `replace`. In `add` mode, documents already in the graph are
  skipped (no wasted re-extraction), new documents are merged into the
  existing graph via the same tiered resolver, and community detection +
  R-GCN training re-run over the *whole* merged graph (cheap — only the LLM
  extraction pass is what's actually skipped for old docs).
- **R-GCN node embeddings** (`app/rgcn.py`) — see below.

### 2. R-GCN, since GraphRAG was losing to vector RAG

Two separate problems were fixed:

**(a) The graph needed better retrieval, not just better embeddings.**
`query_engine.py` now does three things it didn't before:
- Local search walks **2 hops** instead of 1, and additionally pulls in
  nodes that are close in **R-GCN embedding space** even if they aren't
  directly graph-adjacent — this bridges relationships the extraction step
  failed to link explicitly (tagged `[via R-GCN]` in the context so you can
  see when it fires).
- Local search now includes the **actual source chunk text**, not just
  short evidence paraphrases — previously the answer model only ever saw a
  compressed, lossy summary of what the document said, while the vector
  baseline always saw the literal prose. That gap alone was a likely reason
  GraphRAG lost on precise-number questions.
- Global search **ranks community summaries by relevance** to the question
  (via local text embeddings) instead of dumping every summary into the
  prompt — with a small model, irrelevant context is actively harmful, not
  just wasted tokens.

**(b) R-GCN specifically** (`app/rgcn.py`): since there's no labeled data,
it's trained self-supervised via link prediction — a small relational graph
conv net (one weight matrix per *bucketed* relation type, since raw
predicates are free text) learns node embeddings such that real edges score
higher than randomly-corrupted negative pairs under a DistMult decoder.
Training runs automatically after every ingestion (a few seconds on CPU for
graphs this size) and is used for the local-search neighborhood expansion
above. It's intentionally scoped to graph-internal tasks — linking an
arbitrary new query string to a node uses the general-purpose text embedder
in `text_embeddings.py` instead, since R-GCN's embeddings are transductive
(only defined for nodes already in the graph).

If you don't want to install `torch`, none of this is required — the app
falls back to the 2-hop/hybrid-grounding/ranked-global-search improvements
alone, which should already help.

### 3. Drag-and-drop upload

The sidebar's file input is now a real dropzone (`frontend/index.html`) —
drag files onto it or click to browse, with an add-vs-replace toggle once a
corpus already exists.

### 4. Model swap

Default model is now `qwen2.5:3b-instruct` (better JSON/instruction
adherence than `llama3.2:3b` at the same size class). Change it via the
`LEDGER_*_MODEL` env vars above — nothing else in the code assumes a
specific model family.

### 5. Tuned for financial reports

Full pivot, not a bolt-on: entity types are now `Company`, `Metric`,
`Period`, `Segment`, `Person`, `RiskFactor`, `Guidance`, `Other`; the
extraction prompt explicitly ties every numeric metric to its fiscal period
and instructs the model to copy figures exactly rather than round or
paraphrase them; the bundled sample corpus is three real Q2 2025 SEC
earnings releases (MaxLinear — semiconductor, loss-making; Five9 — SaaS,
turned profitable; NVR — homebuilder, margin decline) chosen so there's
real cross-company comparison potential (YoY growth, margin trends,
profitability inflection) and genuine multi-hop numeric questions (e.g.
"how does stock-based comp compare to GAAP net income," which needs two
numbers pulled from different parts of the same filing). The benchmark
question set was rewritten to match.

## Known limitations (still honest, per the PRD's own risk table)

- Tier-2/3 entity resolution depends on the local embedding model and LLM
  judgment quality — it's a real improvement over pure string matching, not
  a guarantee against occasional over/under-merging.
- R-GCN is trained fresh on a small graph every ingestion; it's a genuine
  self-supervised signal, not a large pretrained graph model — treat its
  neighborhood suggestions as one useful signal among several, not ground
  truth.
- The benchmark's LLM judge is only as reliable as the local model doing the
  judging — read `judge_rationale`, not just the score.
- Extraction and answer quality still depend heavily on which local model
  you point at; try a bigger model if the graph looks sparse or answers feel
  shallow.