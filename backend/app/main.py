from __future__ import annotations
import traceback
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .extraction import extract_document, chunk_text
from .graph_store import GraphStore, reset_store, get_store
from .community import build_community_summaries, save_summaries, load_summaries
from .models import (
    IngestResponse, QueryRequest, QueryResponse, CompareResponse,
    BenchmarkSummary,
)
from . import query_engine, vector_baseline, benchmark, ollama_client, rgcn

app = FastAPI(title="Ledger — GraphRAG for Financial Reports")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_text_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(errors="ignore")


def _ingest_documents(doc_paths: List[Path], mode: str = "replace") -> IngestResponse:
    """mode="replace" wipes the existing graph/vector store and rebuilds from
    just doc_paths. mode="add" loads the existing store and merges doc_paths
    into it (skipping any doc_id already ingested, for idempotency), then
    re-runs community detection and R-GCN training over the *whole* merged
    graph — those are cheap; only the LLM extraction pass is skipped for
    already-ingested docs."""
    if mode == "replace":
        store = reset_store()
        vector_baseline.reset_collection()
        already_ingested = set()
    else:
        store = get_store()
        vector_baseline.get_collection()
        already_ingested = store.ingested_doc_ids()

    total_triples = 0
    doc_ids = []
    for path in doc_paths:
        doc_id = path.stem
        doc_ids.append(doc_id)
        if doc_id in already_ingested:
            continue  # already in the graph from a previous "add" — skip re-extraction
        text = _read_text_file(path)

        chunks = chunk_text(text, doc_id)
        extraction_results = extract_document(doc_id, text)
        store.ingest_document_chunks(doc_id, chunks, extraction_results)
        total_triples += sum(len(r.triples) for r in extraction_results)

        vector_baseline.ingest_document(doc_id, text)

    store.save()

    summaries = build_community_summaries(store)
    save_summaries(summaries)

    try:
        rgcn.train_and_save(store.graph)
    except Exception:
        traceback.print_exc()  # R-GCN is a retrieval enhancement, not load-bearing — don't fail ingestion

    return IngestResponse(
        doc_ids=doc_ids,
        num_chunks=len(store.chunks),
        num_entities=store.graph.number_of_nodes(),
        num_triples=total_triples,
        num_communities=len(summaries),
    )


@app.get("/api/status")
def status():
    store = GraphStore.load()
    summaries = load_summaries()
    sample_docs = sorted(p.stem for p in config.SAMPLE_DOCS_DIR.glob("*.txt"))
    ollama_up = ollama_client.is_available()
    rgcn_state = rgcn.load_embeddings()
    return {
        "ingested": store is not None and store.graph.number_of_nodes() > 0,
        "num_nodes": store.graph.number_of_nodes() if store else 0,
        "num_edges": store.graph.number_of_edges() if store else 0,
        "num_communities": len(summaries),
        "doc_ids": sorted(store.ingested_doc_ids()) if store else [],
        "sample_docs_available": sample_docs,
        "ollama_available": ollama_up,
        "ollama_models": ollama_client.list_models() if ollama_up else [],
        "ollama_host": config.OLLAMA_HOST,
        "configured_models": {
            "extraction": config.EXTRACTION_MODEL,
            "answer": config.ANSWER_MODEL,
            "judge": config.JUDGE_MODEL,
        },
        "rgcn_available": rgcn.TORCH_AVAILABLE,
        "rgcn_trained": rgcn_state is not None,
        "rgcn_num_nodes": rgcn_state[1].shape[0] if rgcn_state else 0,
    }


@app.post("/api/ingest/sample", response_model=IngestResponse)
def ingest_sample():
    if not ollama_client.is_available():
        raise HTTPException(400, f"Can't reach Ollama at {config.OLLAMA_HOST}. Run `ollama serve` "
                                  f"and `ollama pull {config.EXTRACTION_MODEL}` first.")
    paths = sorted(config.SAMPLE_DOCS_DIR.glob("*.txt"))
    if not paths:
        raise HTTPException(404, "No sample documents found.")
    try:
        return _ingest_documents(paths, mode="replace")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Ingestion failed: {e}")


@app.post("/api/ingest/upload", response_model=IngestResponse)
async def ingest_upload(files: List[UploadFile] = File(...), mode: str = Form("add")):
    if mode not in ("add", "replace"):
        raise HTTPException(400, "mode must be 'add' or 'replace'")
    if not ollama_client.is_available():
        raise HTTPException(400, f"Can't reach Ollama at {config.OLLAMA_HOST}. Run `ollama serve` "
                                  f"and `ollama pull {config.EXTRACTION_MODEL}` first.")
    config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for f in files:
        dest = config.UPLOADS_DIR / f.filename
        content = await f.read()
        dest.write_bytes(content)
        saved_paths.append(dest)
    try:
        return _ingest_documents(saved_paths, mode=mode)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Ingestion failed: {e}")


@app.get("/api/graph")
def get_graph():
    store = GraphStore.load()
    if store is None:
        return {"nodes": [], "links": []}
    nodes = [
        {"id": n, "type": d.get("type", "Other"), "description": d.get("description", ""),
         "source_docs": d.get("source_docs", []),
         "degree": store.graph.degree(n)}
        for n, d in store.graph.nodes(data=True)
    ]
    links = [
        {"source": u, "target": v, "predicate": d.get("predicate", ""), "doc_id": d.get("doc_id", "")}
        for u, v, d in store.graph.edges(data=True)
    ]
    return {"nodes": nodes, "links": links}


@app.get("/api/communities")
def get_communities():
    return [s.model_dump() for s in load_summaries()]


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        return query_engine.answer_question(req.question, mode=req.mode)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/api/query/compare", response_model=CompareResponse)
def query_compare(req: QueryRequest):
    try:
        graphrag_resp = query_engine.answer_question(req.question, mode=req.mode)
        vector_resp = vector_baseline.answer_question(req.question)
        return CompareResponse(question=req.question, graphrag=graphrag_resp, vector_rag=vector_resp)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.get("/api/benchmark/questions")
def get_benchmark_questions():
    return [q.model_dump() for q in benchmark.load_questions()]


@app.post("/api/benchmark/run", response_model=BenchmarkSummary)
def run_benchmark():
    try:
        return benchmark.run_benchmark()
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.get("/api/benchmark/results")
def get_benchmark_results():
    summary = benchmark.load_last_results()
    if summary is None:
        return {"results": [], "graphrag_avg": None, "vector_rag_avg": None}
    return summary.model_dump()