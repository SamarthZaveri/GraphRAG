"""Background ingestion jobs.

LLM extraction takes minutes per document on a local 14B model, far beyond an
HTTP timeout, so ingestion runs on a single worker thread (one job at a time,
matching Ollama's single-request concurrency) and clients poll its progress."""
from __future__ import annotations
import contextvars
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")
_jobs: dict = {}
_lock = threading.Lock()
MAX_KEPT_JOBS = 50


def _update(job_id, **fields):
    with _lock:
        _jobs[job_id].update(fields)


def submit(kind, total_docs, fn, owner=None):
    """Queue fn(progress) where progress(done, current_doc) reports per-document progress.
    The job runs in a copy of the caller's context, so it ingests into the caller's workspace."""
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[job_id] = {"id": job_id, "kind": kind, "owner": owner, "status": "queued", "created": time.time(),
                         "started": None, "finished": None, "progress": {"done": 0, "total": total_docs, "current": None},
                         "result": None, "error": None}
        for old in sorted(_jobs.values(), key=lambda j: j["created"])[:-MAX_KEPT_JOBS]:
            if old["status"] in ("done", "error"):
                del _jobs[old["id"]]

    def progress(done, current=None):
        _update(job_id, progress={"done": done, "total": total_docs, "current": current})

    def run():
        _update(job_id, status="running", started=time.time())
        try:
            result = fn(progress)
            _update(job_id, status="done", finished=time.time(),
                    result=result.model_dump() if hasattr(result, "model_dump") else result)
        except Exception as error:
            traceback.print_exc()
            _update(job_id, status="error", finished=time.time(), error=getattr(error, "detail", None) or str(error))

    _executor.submit(contextvars.copy_context().run, run)
    return get(job_id, owner)


def get(job_id, owner=None):
    with _lock:
        job = _jobs.get(job_id)
        if job is None or job["owner"] != owner:
            return None
        ahead = sum(1 for j in _jobs.values() if j["status"] in ("queued", "running") and j["created"] < job["created"])
        return {**job, "progress": dict(job["progress"]), "jobs_ahead": ahead}


def active(owner=None):
    with _lock:
        return [dict(j) for j in _jobs.values() if j["status"] in ("queued", "running") and j["owner"] == owner]
