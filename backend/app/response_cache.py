"""Answers already given for a question on the same corpus, served without calling the LLM.

The key covers everything that decides an answer: the normalized question, the requested engine
and mode, the corpus content (hashes of the shared corpus's or the caller's workspace graph-state
files), the router policy, the models and the retrieval version. Re-ingesting changes the corpus
hashes, so a stale answer is never served; old rows are dropped beyond RESPONSE_CACHE_MAX_ROWS.
Content hashes (not file times) keep a cache valid when a pre-ingested corpus is copied to
another machine together with its cache file."""
from __future__ import annotations
import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

from . import config, workspace

CORPUS_FILES = ("meta.json", "graph.gpickle", "chunks.json", "community_summaries.json")
_lock = threading.Lock()
_digests = {}  # path -> ((mtime_ns, size), sha256), so unchanged files are hashed once


def _digest(path: Path) -> Optional[str]:
    try:
        st = path.stat()
    except OSError:
        return None
    stamp = (st.st_mtime_ns, st.st_size)
    cached = _digests.get(path)
    if cached and cached[0] == stamp:
        return cached[1]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    _digests[path] = (stamp, digest)
    return digest


def corpus_fingerprint() -> Optional[str]:
    """None when nothing has been ingested (no cache for an empty corpus)."""
    from .corpus_router import QUERY_BANDIT_PATH
    state = workspace.state_dir()
    files = {name: _digest(state / name) for name in CORPUS_FILES}
    if files["meta.json"] is None:
        return None
    files["policy"] = _digest(QUERY_BANDIT_PATH)
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def key(question: str, engine: str, mode: str, kind: str = "query") -> Optional[str]:
    if not config.RESPONSE_CACHE:
        return None
    corpus = corpus_fingerprint()
    if corpus is None:
        return None
    return hashlib.sha256(json.dumps({
        "kind": kind, "question": " ".join(question.split()).lower(), "engine": engine, "mode": mode,
        "corpus": corpus, "retrieval_version": config.RETRIEVAL_VERSION, "llm_backend": config.LLM_BACKEND,
        "models": [config.ANSWER_MODEL, config.QUERY_MODEL, config.EXTRACTION_MODEL],
    }, sort_keys=True).encode()).hexdigest()


def _connect():
    conn = sqlite3.connect(config.RESPONSE_CACHE_PATH, timeout=10)
    conn.execute("CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, ts REAL NOT NULL, payload TEXT NOT NULL)")
    return conn


def get(cache_key: Optional[str]) -> Optional[dict]:
    if cache_key is None:
        return None
    try:
        with _lock:
            conn = _connect()
            try:
                row = conn.execute("SELECT payload FROM answers WHERE key = ?", (cache_key,)).fetchone()
            finally:
                conn.close()
        return json.loads(row[0]) if row else None
    except (sqlite3.Error, ValueError) as error:
        print(f"  [cache] read failed: {error}", flush=True)
        return None


def put(cache_key: Optional[str], payload: dict):
    """A cache failure must never fail the request that produced the answer."""
    if cache_key is None:
        return
    try:
        with _lock:
            conn = _connect()
            try:
                with conn:
                    conn.execute("INSERT OR REPLACE INTO answers (key, ts, payload) VALUES (?, ?, ?)",
                                 (cache_key, time.time(), json.dumps(payload)))
                    conn.execute("DELETE FROM answers WHERE key NOT IN (SELECT key FROM answers ORDER BY ts DESC LIMIT ?)",
                                 (config.RESPONSE_CACHE_MAX_ROWS,))
            finally:
                conn.close()
    except sqlite3.Error as error:
        print(f"  [cache] write failed: {error}", flush=True)
