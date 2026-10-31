---
title: Ledger Adaptive RAG
emoji: 📒
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Three retrievers and a learned router over SEC filings
---

# Ledger: adaptive retrieval over financial filings

Three retrieval engines (hybrid BM25 + dense, dense vector, knowledge graph) answer questions over SEC
earnings releases, and a LinUCB router trained on judged rewards picks one per question.

This Space runs everything on a free CPU: `qwen2.5:7b-instruct` through Ollama, the FastAPI backend
and the frontend. Expect roughly a minute per answer. The benchmark results and recorded answers from
all three engines load instantly.

Code, benchmark and write-up: https://github.com/SamarthZaveri/GraphRAG

## Space settings

| Variable | Value |
|---|---|
| `SEED_URL` | URL of `demo_seed.zip` (GitHub release asset made by `scripts/build_demo_corpus.ps1`) |
| `LEDGER_CORS_ORIGINS` | your Vercel URL, e.g. `https://ledger-rag.vercel.app` |
| `LEDGER_ADMIN_TOKEN` (secret) | optional; enables admin-only endpoints |
