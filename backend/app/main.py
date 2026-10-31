from __future__ import annotations
import contextvars
import queue
import threading
import time
import traceback
from contextlib import ExitStack
from pathlib import Path
from typing import List, Optional
import json

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from . import config
from .extraction import extract_document
from .graph_store import GraphStore, reset_store, get_store
from .community import build_community_summaries, save_summaries, load_summaries, load_modularity
from .models import (
    IngestResponse, QueryRequest, QueryResponse, CompareResponse,
    BenchmarkSummary,
)
from . import vector_baseline, benchmark, llm_client, rgcn, corpus_router, engines, telemetry, metrics_store, jobs
from . import guard, workspace, answering, response_cache
import re

app = FastAPI(title="Ledger — Adaptive RAG Router for Financial Reports")

# Starlette runs the last-added middleware first: CORS wraps the guard, so 429s carry CORS headers too.
app.add_middleware(guard.GuardMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Retry-After", "X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Group"],
)
_admin_only = [Depends(guard.require_admin_in_public_mode)]


def _read_text_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def _ingest_documents(doc_paths: List[Path], mode: str = "replace", progress=None) -> IngestResponse:
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
    for index, path in enumerate(doc_paths):
        doc_id = path.stem
        doc_ids.append(doc_id)
        if progress:
            progress(index, doc_id)
        if doc_id in already_ingested:
            continue  # already in the graph from a previous "add" — skip re-extraction
        text = _read_text_file(path)

        # Table-aware extraction: deterministic table facts + concurrent LLM
        # extraction over the remaining prose (see extraction.py / table_parser.py)
        chunks, extraction_results = extract_document(doc_id, text)
        store.ingest_document_chunks(doc_id, chunks, extraction_results)
        total_triples += sum(len(r.triples) for r in extraction_results)

        vector_baseline.ingest_document(doc_id, text)

    store.save()
    if progress:
        progress(len(doc_paths), "building communities and router features")

    summaries = build_community_summaries(store)
    save_summaries(summaries)

    try:
        rgcn.train_and_save(store.graph)
    except Exception:
        traceback.print_exc()  # R-GCN is a retrieval enhancement, not load-bearing — don't fail ingestion

    try:
        corpus_router.analyze_corpus()
    except Exception:
        traceback.print_exc()  # router recommendation isn't load-bearing either

    return IngestResponse(
        doc_ids=doc_ids,
        num_chunks=len(store.chunks),
        num_entities=store.graph.number_of_nodes(),
        num_triples=total_triples,
        num_communities=len(summaries),
    )


def _traced_ingest(endpoint: str, paths: List[Path], mode: str, progress=None) -> IngestResponse:
    """Ingest with extraction cost attached: this is the up-front price GraphRAG pays per document."""
    with telemetry.trace() as t:
        try:
            with telemetry.span("ingestion"):
                result = _ingest_documents(paths, mode=mode, progress=progress)
        except Exception as e:
            traceback.print_exc()
            metrics_store.record(endpoint, "error", t.summary(), detail={"error": str(e)[:300], "docs": len(paths)})
            raise RuntimeError(f"Ingestion failed: {e}") from e
        result.metrics = t.summary()
        new_docs = max(1, len(paths))
        result.metrics["per_document"] = {
            "latency_ms": round(result.metrics["latency_ms"] / new_docs, 1),
            "tokens": round(result.metrics["total_tokens"] / new_docs, 1),
            "cost_usd": round(result.metrics["cost_usd"] / new_docs, 6),
        }
        metrics_store.record(endpoint, "ok", result.metrics,
                             detail={"docs": len(paths), "chunks": result.num_chunks, "triples": result.num_triples})
        return result


@app.get("/api/health")
def health():
    """Cheap liveness probe for load balancers; does not touch the graph or the LLM."""
    return {"status": "ok", "retrieval_version": config.RETRIEVAL_VERSION, "inference_device": config.INFERENCE_DEVICE}


@app.get("/api/status")
def status(request: Request):
    """Shared-corpus state and serving configuration, plus the caller's private workspace (if any)."""
    workspace.cleanup()
    ws_id = request.headers.get(guard.WORKSPACE_HEADER)
    ws_info = workspace.info(ws_id) if workspace.valid_id(ws_id) else None
    with workspace.use(None):
        return {**_corpus_status(), "workspace": ws_info,
                "active_jobs": jobs.active(ws_id if ws_info else None),
                "public_mode": config.PUBLIC_MODE,
                "inference_device": config.INFERENCE_DEVICE,
                "response_cache": config.RESPONSE_CACHE,
                "rate_limits": config.RATE_LIMITS}


def _corpus_status():
    store = GraphStore.load()
    summaries = load_summaries()
    sample_docs = sorted(p.stem for p in config.SAMPLE_DOCS_DIR.glob("*.txt"))
    ollama_up = llm_client.is_available()
    rgcn_state = rgcn.load_embeddings()
    rgcn_metrics = rgcn.load_metrics()
    modularity_info = load_modularity()
    return {
        "ingested": store is not None and store.graph.number_of_nodes() > 0,
        "num_nodes": store.graph.number_of_nodes() if store else 0,
        "num_edges": store.graph.number_of_edges() if store else 0,
        "num_communities": len(summaries),
        "doc_ids": sorted(store.ingested_doc_ids()) if store else [],
        "sample_docs_available": sample_docs,
        "ollama_available": ollama_up,
        "ollama_models": llm_client.list_models() if ollama_up else [],
        "ollama_host": config.OLLAMA_HOST,
        "llm_backend": config.LLM_BACKEND,
        "llm_available": ollama_up,
        "llm_endpoint": config.API_BASE_URL if config.LLM_BACKEND == "api" else config.OLLAMA_HOST,
        "configured_models": {
            "extraction": config.EXTRACTION_MODEL,
            "query": config.QUERY_MODEL,
            "answer": config.ANSWER_MODEL,
            "judge": config.JUDGE_MODEL,
            "question": config.QUESTION_MODEL,
        },
        "rgcn_available": rgcn.TORCH_AVAILABLE,
        "rgcn_trained": rgcn_state is not None,
        "rgcn_num_nodes": rgcn_state[1].shape[0] if rgcn_state else 0,
        "rgcn_val_auc": rgcn_metrics.get("val_auc") if rgcn_metrics else None,
        "rgcn_used_semantic_init": rgcn_metrics.get("used_semantic_init") if rgcn_metrics else None,
        "community_modularity": modularity_info.get("modularity"),
        "router": corpus_router.policy_status(),
        "upload_limits": {"max_files": config.MAX_UPLOAD_FILES, "max_mb": config.MAX_UPLOAD_MB,
                          "max_pdf_pages": config.MAX_PDF_PAGES,
                          "extensions": list(config.ALLOWED_UPLOAD_EXTENSIONS),
                          "workspace_max_docs": config.MAX_WORKSPACE_DOCS,
                          "workspace_max_chars": config.MAX_WORKSPACE_CHARS,
                          "workspace_ttl_hours": config.WORKSPACE_TTL_HOURS},
    }


_LLM_DOWN = "LLM endpoint unavailable. Check HF_TOKEN/API configuration or start your configured Ollama server."


@app.post("/api/ingest/sample", dependencies=_admin_only)
def ingest_sample():
    """Queue ingestion of the bundled sample corpus into the shared corpus; poll /api/ingest/jobs/{id}."""
    if workspace.current():
        raise HTTPException(400, "The sample corpus loads into the shared corpus, not a private workspace.")
    if not llm_client.is_available():
        raise HTTPException(400, _LLM_DOWN)
    paths = sorted(config.SAMPLE_DOCS_DIR.glob("*.txt"))
    if not paths:
        raise HTTPException(404, "No sample documents found.")
    return jobs.submit("sample", len(paths), lambda progress: _traced_ingest("ingest_sample", paths, "replace", progress))


def _safe_filename(name: str) -> str:
    # Drop any client-supplied directories and characters that could escape UPLOADS_DIR.
    stem, suffix = Path(Path(name or "").name).stem, Path(name or "").suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")[:80] or "document"
    return stem + suffix


async def _read_limited(f: UploadFile, limit: int) -> bytes:
    content = bytearray()
    while chunk := await f.read(1024 * 1024):
        content.extend(chunk)
        if len(content) > limit:
            raise HTTPException(413, f"{f.filename} exceeds the {config.MAX_UPLOAD_MB:g} MB limit.")
    return bytes(content)


def _validate_upload(name: str, content: bytes) -> int:
    """Reject unusable files; returns the extracted text length (for workspace quotas)."""
    if not content.strip():
        raise HTTPException(422, f"{name} is empty.")
    if name.endswith(".pdf"):
        if not content.startswith(b"%PDF"):
            raise HTTPException(415, f"{name} is not a valid PDF.")
        try:
            from io import BytesIO
            from pypdf import PdfReader
            reader = PdfReader(BytesIO(content))
            if reader.is_encrypted:
                raise HTTPException(422, f"{name} is encrypted.")
            if len(reader.pages) > config.MAX_PDF_PAGES:
                raise HTTPException(413, f"{name} has {len(reader.pages)} pages; the limit is {config.MAX_PDF_PAGES}. "
                                         "Upload the relevant section (e.g. the earnings release, not the full annual report).")
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            if not text.strip():
                raise HTTPException(422, f"{name} has no extractable text (scanned PDF?).")
            return len(text)
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(422, f"{name} could not be parsed as a PDF.")
    else:
        try:
            return len(content.decode("utf-8"))
        except UnicodeDecodeError:
            raise HTTPException(415, f"{name} is not UTF-8 text.")


@app.post("/api/ingest/upload")
async def ingest_upload(request: Request, files: List[UploadFile] = File(...), mode: str = Form("add")):
    """Validate uploads (count, size, type, parseability, workspace quota), then queue ingestion as a
    background job. With an X-Ledger-Workspace header, files go to that private workspace; without one
    they go to the shared corpus, which is admin-only on the public deployment."""
    ws = workspace.current()
    if ws is None:
        guard.require_admin_in_public_mode(request)
    if mode not in ("add", "replace"):
        raise HTTPException(400, "mode must be 'add' or 'replace'")
    if len(files) > config.MAX_UPLOAD_FILES:
        raise HTTPException(413, f"Upload at most {config.MAX_UPLOAD_FILES} files at a time.")
    if not llm_client.is_available():
        raise HTTPException(400, _LLM_DOWN)
    limit = int(config.MAX_UPLOAD_MB * 1024 * 1024)
    staged = []
    for f in files:
        name = _safe_filename(f.filename)
        if Path(name).suffix not in config.ALLOWED_UPLOAD_EXTENSIONS:
            raise HTTPException(415, f"{f.filename}: allowed types are {', '.join(config.ALLOWED_UPLOAD_EXTENSIONS)}.")
        content = await _read_limited(f, limit)
        staged.append((name, content, _validate_upload(name, content)))
    if ws is not None:
        chars = _check_workspace_quota(ws, staged, mode)
    # Write only after every file passed, so a rejected batch leaves nothing behind.
    uploads = workspace.uploads_dir()
    if ws is not None and mode == "replace":
        for old in uploads.glob("*"):
            old.unlink(missing_ok=True)
    uploads.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for name, content, _ in staged:
        dest = uploads / name
        dest.write_bytes(content)
        saved_paths.append(dest)
    if ws is not None:
        workspace.touch(ws, chars=chars)
    return jobs.submit("upload", len(saved_paths),
                       lambda progress: _traced_ingest("ingest_upload", saved_paths, mode, progress), owner=ws)


def _check_workspace_quota(ws, staged, mode) -> int:
    """Bound a workspace's documents and text (extraction cost scales with text). Returns the new total chars."""
    current = workspace.info(ws) if mode == "add" else None
    existing = set(current["docs"]) if current else set()
    docs = existing | {Path(name).stem for name, _, _ in staged}
    if len(docs) > config.MAX_WORKSPACE_DOCS:
        raise HTTPException(413, f"A workspace holds at most {config.MAX_WORKSPACE_DOCS} documents "
                                 f"({len(existing)} already uploaded). Use 'replace' to start over.")
    chars = (current["chars"] if current else 0) + sum(c for _, _, c in staged)
    if chars > config.MAX_WORKSPACE_CHARS:
        raise HTTPException(413, f"These files contain {chars:,} characters of text; a workspace allows "
                                 f"{config.MAX_WORKSPACE_CHARS:,}. Upload fewer or shorter documents.")
    return chars


@app.delete("/api/workspace")
def delete_workspace():
    """Let a visitor wipe their private workspace immediately."""
    ws = workspace.current()
    if ws is None:
        raise HTTPException(404, "No workspace.")
    if jobs.active(ws):
        raise HTTPException(409, "An ingestion job is still running for this workspace.")
    workspace.delete(ws)
    return {"deleted": ws}


@app.get("/api/ingest/jobs/{job_id}")
def get_ingest_job(job_id: str):
    job = jobs.get(job_id, workspace.current())
    if job is None:
        raise HTTPException(404, "Unknown or expired job.")
    return job


@app.get("/api/graph", dependencies=_admin_only)
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


@app.get("/api/communities", dependencies=_admin_only)
def get_communities():
    return [s.model_dump() for s in load_summaries()]


@app.get("/api/corpus-analysis", dependencies=_admin_only)
def get_corpus_analysis():
    analysis = corpus_router.load_analysis()
    if analysis is None:
        return corpus_router.analyze_corpus()
    return analysis


def _traced_answer(engine: str, question: str, mode: str, endpoint: str, routing: dict = None, t=None) -> QueryResponse:
    """Answer with one engine and attach latency/token/cost metrics and quality signals.
    Pass an open trace `t` to include work done before this call (e.g. routing)."""
    def run(trace):
        classification = (routing["category"], routing.get("entities") or []) if routing and routing.get("category") else None
        with telemetry.span("engine"):
            resp = engines.answer_question(engine, question, mode=mode, classification=classification)
        resp.metrics = trace.summary()
        engine_ms = resp.metrics["stage_ms"].get("engine", 0.0)
        engine_llm_ms = sum(s["ms"] for stage, s in resp.metrics["llm_by_stage"].items() if stage != "routing")
        # Non-generation engine time: retrieval, ranking, graph traversal and query analysis.
        resp.metrics["stage_ms"]["retrieval"] = round(engine_ms - resp.metrics["stage_ms"].get("generation", 0.0), 1)
        resp.metrics["retrieval_llm_ms"] = round(engine_llm_ms - resp.metrics["stage_ms"].get("generation", 0.0), 1)
        resp.quality = telemetry.answer_quality(resp.answer, resp.citations)
        resp.routing = routing
        metrics_store.record(endpoint, "ok", resp.metrics, engine=engine, routed=routing is not None,
                             quality=resp.quality, detail={"mode_used": resp.mode_used})
        return resp
    if t is not None:
        return run(t)
    with telemetry.trace() as trace:
        return run(trace)


def _touch_workspace():
    if workspace.current():
        workspace.touch(workspace.current())  # active use extends the expiry


def _answer(req: QueryRequest, t, endpoint: str = "query", on_route=None) -> QueryResponse:
    """Route (engine="auto") and answer inside the open trace `t`."""
    engine, routing = req.engine, None
    if engine == "auto":
        with telemetry.span("routing"):
            routing = corpus_router.route_query_details(req.question)
        engine = routing["engine"]
    if on_route:
        on_route(engine, routing)
    resp = _traced_answer(engine, req.question, req.mode, endpoint, routing, t)
    resp.engine_used = engine
    resp.engine_reason = routing["reason"] if routing else None
    return resp


def _cached(cache_key, endpoint, model):
    """A stored answer for this question and corpus, recorded as a cache hit; None on a miss."""
    started = time.perf_counter()
    payload = response_cache.get(cache_key)
    if payload is None:
        return None
    resp = model(**payload)
    resp.cache_hit = True
    engine = getattr(resp, "engine_used", None)
    metrics_store.record(endpoint, "cached", {"latency_ms": round((time.perf_counter() - started) * 1000, 1)},
                         engine=engine, routed=getattr(resp, "routing", None) is not None)
    return resp


def _cacheable(resp: QueryResponse) -> bool:
    return resp.answer != answering.TRUNCATED_ANSWER


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Single-answer query for the Ask tab. engine="auto" now routes PER
    QUESTION via corpus_router.route_query(), which combines this corpus's
    doc-level structure with a live classification of the question itself
    (local/global/multi_hop/conflict) and scores all three engines using a
    trained contextual bandit. This replaced a static, corpus-wide
    recommendation after real evaluation data showed corpus structure
    alone doesn't predict which engine wins on any individual question --
    see corpus_router.route_query's docstring. Falls back automatically to
    the previous rule-based corpus-level recommendation if no trained
    query bandit is available yet. "graphrag"/"vector_rag" still force a
    manual choice, unchanged. This routing does NOT apply to
    /api/query/compare, which always runs all three engines by design.
    Repeated questions on an unchanged corpus are answered from response_cache."""
    _touch_workspace()
    cache_key = response_cache.key(req.question, req.engine, req.mode)
    hit = _cached(cache_key, "query", QueryResponse)
    if hit is not None:
        return hit
    with guard.llm_slot(), telemetry.trace() as t:
        try:
            resp = _answer(req, t)
            if _cacheable(resp):
                response_cache.put(cache_key, resp.model_dump())
            return resp
        except HTTPException:
            raise
        except Exception as e:
            traceback.print_exc()
            metrics_store.record("query", "error", t.summary(), engine=req.engine, detail={"error": str(e)[:300]})
            raise HTTPException(500, str(e))


@app.post("/api/query/compare", response_model=CompareResponse)
def query_compare(req: QueryRequest):
    # Each engine runs under its own trace so per-engine latency/tokens/cost are comparable.
    _touch_workspace()
    cache_key = response_cache.key(req.question, "compare", req.mode, kind="compare")
    hit = _cached(cache_key, "compare", CompareResponse)
    if hit is not None:
        return hit
    try:
        with guard.llm_slot():
            answers = {e: _traced_answer(e, req.question, req.mode, "compare") for e in engines.ENGINE_NAMES}
        resp = CompareResponse(question=req.question, graphrag=answers["graphrag"],
                               vector_rag=answers["vector_rag"], hybrid_rag=answers["hybrid_rag"])
        if all(_cacheable(a) for a in answers.values()):
            response_cache.put(cache_key, resp.model_dump())
        return resp
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        metrics_store.record("compare", "error", detail={"error": str(e)[:300]})
        raise HTTPException(500, str(e))


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
SSE_KEEPALIVE_SECONDS = 15


@app.post("/api/query/stream")
def query_stream(req: QueryRequest):
    """/api/query as server-sent events, so the answer appears as it is generated (on a CPU host a
    full answer takes minutes). Events: `meta` {engine, routing} once routing is decided; `token`
    (answer text); `reset` (discard the text so far: regenerating with a larger budget); then
    `done` with the full QueryResponse (metrics include ttft_ms), or `error` {detail}. Comment
    lines keep idle proxies from closing the connection during retrieval. A cached answer is sent
    as a single `done`."""
    _touch_workspace()
    cache_key = response_cache.key(req.question, req.engine, req.mode)
    hit = _cached(cache_key, "query_stream", QueryResponse)
    if hit is not None:
        return StreamingResponse(iter([_sse("done", hit.model_dump())]), media_type="text/event-stream", headers=SSE_HEADERS)
    slot = ExitStack()
    slot.enter_context(guard.llm_slot())  # a busy model is a plain 503, before the stream starts
    events = queue.Queue()

    def work():
        try:
            with answering.streaming(lambda piece: events.put(("token", piece) if piece is not None else ("reset", ""))), \
                    telemetry.trace() as t:
                try:
                    resp = _answer(req, t, "query_stream",
                                   on_route=lambda engine, routing: events.put(("meta", {"engine": engine, "routing": routing})))
                except Exception as e:
                    metrics_store.record("query_stream", "error", t.summary(), engine=req.engine, detail={"error": str(e)[:300]})
                    raise
            if _cacheable(resp):
                response_cache.put(cache_key, resp.model_dump())
            events.put(("done", resp.model_dump()))
        except Exception as e:
            traceback.print_exc()
            events.put(("error", {"detail": str(e)[:300]}))
        finally:
            slot.close()
            events.put(None)

    # The worker carries this request's context (workspace binding); it finishes and fills the
    # cache even if the client disconnects.
    threading.Thread(target=contextvars.copy_context().run, args=(work,), daemon=True).start()

    def stream():
        while True:
            try:
                item = events.get(timeout=SSE_KEEPALIVE_SECONDS)
            except queue.Empty:
                yield ": keep-alive\n\n"
                continue
            if item is None:
                return
            yield _sse(*item)

    return StreamingResponse(stream(), media_type="text/event-stream", headers=SSE_HEADERS)


@app.get("/api/benchmark/questions", dependencies=_admin_only)
def get_benchmark_questions():
    return [q.model_dump() for q in benchmark.load_questions()]


@app.post("/api/benchmark/run", response_model=BenchmarkSummary, dependencies=_admin_only)
def run_benchmark():
    try:
        return benchmark.run_benchmark()
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.get("/api/benchmark/results", dependencies=_admin_only)
def get_benchmark_results():
    summary = benchmark.load_last_results()
    if summary is None:
        return {"results": [], "graphrag_avg": None, "vector_rag_avg": None}
    return summary.model_dump()

@app.get("/api/metrics")
def get_metrics(window_minutes: int = Query(60, ge=1, le=60 * 24 * 30), endpoint: Optional[str] = None):
    """Aggregates over recorded requests: latency percentiles (total and per stage), throughput,
    tokens, cost, error rate, abstention/citation-coverage and engine usage."""
    return metrics_store.summary(window_minutes, endpoint)


@app.get("/api/metrics/recent")
def get_recent_metrics(limit: int = Query(50, ge=1, le=500)):
    return metrics_store.recent(limit)


@app.get("/api/rl/results")
def get_rl_results():
    """Offline router evaluation: per-split summaries, analyses (RL/analyze_results.py) and policy metadata."""
    root = config.RL_RESULTS_DIR
    def load(path):
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return None
    policy = load(root / "policy.json")
    splits = ("train", "dev", "test")
    def pair(base, split):
        return {"summary": load(base / split / "summary.json"), "analysis": load(base / split / "analysis.json")}
    # <run>/<dataset>/<split>/ holds per-dataset results; <run>/<split>/ holds the combined ones.
    datasets = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name not in splits
                      and any((p / s / "summary.json").exists() for s in splits)) if root.is_dir() else []
    return {
        "splits": {split: pair(root, split) for split in splits},
        "datasets": {name: {split: pair(root / name, split) for split in splits} for name in datasets},
        "policy": {"arms": policy.get("arms"), "alpha": policy.get("alpha"), "context_dim": policy.get("context_dim"),
                   "metadata": policy.get("metadata")} if policy else None,
    }


@app.get("/api/questions/suggested")
def get_suggested_questions():
    """Questions pre-answered by scripts/warm_cache.py for the shared corpus (instant from the cache)."""
    try:
        return json.loads((config.DATA_DIR / "suggested_questions.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


@app.get("/api/rl/examples")
def get_rl_examples(limit: int = Query(9, ge=1, le=40)):
    """Real routed questions for the landing page: the frozen policy's predicted reward per engine
    (recomputed from each record's saved context) next to the judged reward each engine earned.
    Prefers held-out splits; train examples are in-sample for the policy and flagged as such."""
    from .query_bandit import LinUCBBandit
    root = config.RL_RESULTS_DIR
    try:
        policy = LinUCBBandit.load(root / "policy.json")
    except (OSError, ValueError, KeyError):
        return {"split": None, "held_out": False, "examples": []}
    for split in ("test", "dev", "train"):
        try:
            records = json.loads((root / split / "results.json").read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        records = [r for r in records if len(r.get("context") or []) == policy.context_dim]
        if records:
            break
    else:
        return {"split": None, "held_out": False, "examples": []}
    # Round-robin over categories, preferring text questions (they read better than bare numeric prompts).
    by_cat = {}
    for r in sorted(records, key=lambda r: (r.get("task") == "numeric", len(r["question"]))):
        if 30 <= len(r["question"]) <= 170:
            by_cat.setdefault(r.get("benchmark_category", r["category"]), []).append(r)
    picked = []
    while len(picked) < limit and any(by_cat.values()):
        for queue in by_cat.values():
            if queue and len(picked) < limit:
                picked.append(queue.pop(0))
    examples = []
    for r in picked:
        predicted = {arm: round(policy.predicted_reward(arm, r["context"]), 4) for arm in policy.arms}
        chosen = r.get("router_engine") or max(predicted, key=predicted.get)
        rewards = {e: r.get("reward_" + e) for e in engines.ENGINE_NAMES}
        examples.append({"question": r["question"], "category": r.get("benchmark_category", r["category"]),
                         "task": r.get("task"), "predicted_rewards": predicted, "chosen": chosen,
                         "judged_rewards": rewards, "router_reward": rewards.get(chosen),
                         "oracle_reward": max(v for v in rewards.values() if v is not None)})
    return {"split": split, "held_out": split != "train", "examples": examples}


# Serve the existing static interface from the same origin in deployments.
from fastapi.staticfiles import StaticFiles
_frontend = config.BACKEND_ROOT.parent / "frontend"
if _frontend.exists():
    app.mount("/", StaticFiles(directory=str(_frontend), html=True), name="frontend")
