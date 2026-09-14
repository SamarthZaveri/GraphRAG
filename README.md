# Ledger — GraphRAG vs Vector RAG for Financial Reports

A GraphRAG system for financial-report question-answering, rigorously
benchmarked against a vector-RAG baseline across 12 real-world SEC filing
corpora. Runs fully offline against a local Ollama model — no external API
dependency.

## The finding

Vector RAG outperforms GraphRAG overall on this benchmark — but the
advantage isn't uniform, and treating it as uniform would be the wrong
conclusion. Corpus-level structure (how many documents, how much they
overlap) turned out not to predict which engine wins; what predicts it is
the *type of question being asked*. A contextual bandit router learns this
directly from evaluation data and picks the better engine per question,
rather than defaulting to one engine for an entire corpus.

This is the actual point of the project: not to prove GraphRAG wins, but to
build the infrastructure to find out, and to act on the answer honestly.

## Architecture

```
backend/
  app/
    main.py             FastAPI app — ingest, query, benchmark, compare, graph
    extraction.py        chunking + deterministic table extraction + concurrent
                        LLM entity/relation extraction
    table_parser.py      deterministic regex-based financial table parsing
    graph_store.py       NetworkX graph + tiered entity resolution
    community.py         multi-resolution Leiden + hierarchical summaries
    rgcn.py              self-supervised R-GCN node embeddings
    query_engine.py      GraphRAG's local/global search, hybrid grounding, and
                        a query classifier (local / global / multi_hop / conflict)
    vector_baseline.py    the vector-RAG comparison system
    router_features.py    corpus structure features + per-question context vector
    query_bandit.py       LinUCB contextual bandit (training + live serving)
    corpus_router.py      the live per-question engine routing decision
    benchmark.py          head-to-head LLM-judge scoring
frontend/
  index.html              single static file, no build step, graph visualization,
                         light/dark theme
RL/
  fetch_corpora.py         real SEC EDGAR fetcher
  generate_benchmark.py     document-grounded, typed benchmark question generation
  run_experiments.py        ingestion + benchmark + judge pipeline
  run_ablation.py           isolates GraphRAG's graph retrieval from its
                          vector-similarity fallback
  train_query_bandit.py     trains the per-question router
```

## Why a contextual bandit, and why per-question

An early corpus-level router (pick one engine per corpus, based on structural
features like cross-document entity overlap) was tried first and rejected on
evidence: the two corpora with the *most* cross-document structure in the
12-corpus test set were both won by vector RAG. Structure doesn't predict
outcome. What does is the question itself — a simple lookup question favors
vector search regardless of how complex the underlying corpus is; a
cross-document consistency check is exactly what graph structure is for.

The router therefore classifies each incoming question (local / global /
multi_hop / conflict — the same taxonomy used to generate the benchmark) and
combines that with the corpus's structural features into a single context
vector. A LinUCB bandit, trained on real judged question-answer pairs, scores
both engines and picks the higher-predicted one. LinUCB was chosen
specifically because it's well-suited to small data (dozens to low hundreds
of examples) — it maintains an explicit uncertainty estimate per arm rather
than requiring enough data to train a neural policy safely.

## Results

Across the full evaluation (12 corpora, 130+ benchmark questions, real SEC
filings spanning airlines, banks, retail, semiconductors, SaaS, and
single-company longitudinal filings):

- **Vector RAG wins on raw average score across most corpora.** It's
  consistently strong on direct fact lookups, which dominate real-world
  financial Q&A.
- **GraphRAG's advantage concentrates in specific, identifiable cases**:
  small single-document corpora, and — per the trained router — questions
  that ask whether reported figures are internally consistent (`conflict`
  category). This is a meaningful, non-obvious signal to have discovered
  rather than assumed: graph structure earns its cost on cross-referencing
  work, not on retrieval of an isolated number.
- **A targeted fix to entity resolution meaningfully closed the gap.**
  Large graphs with many similarly-named period entities (e.g. "Q2 2025" vs
  "Q2 2026") occasionally caused GraphRAG to ground answers in the wrong
  fiscal period. Adding a year-conflict penalty to entity matching, plus
  reordering context so vector-grounded content takes precedence,
  meaningfully improved GraphRAG's results across the majority of tested
  corpora — including flipping its single worst-performing corpus in the
  entire project into a win.

The router reflects this: it recommends vector RAG by default and GraphRAG
for the cases where the evidence supports it, rather than hardcoding either
assumption.

## Running it

```powershell
# Ingest and benchmark a corpus (or all of them)
cd RL
python run_experiments.py --corpus <name>

# Train the per-question router
python train_query_bandit.py

# Run the app
cd ..\backend
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## Roadmap

- Expand benchmark coverage for underrepresented question categories
  (`global`, `multi_hop`, `conflict` currently have fewer examples than
  `local`) to strengthen the router's confidence outside simple lookups.
- Restrict the R-GCN's own training objective to cross-document edges, so
  its validation AUC becomes a usable router feature rather than saturating
  uniformly across corpora.
- Deployment: hosting, live metrics, and a production frontend.
