"""Durable per-request metrics (SQLite) and the aggregates served by /api/metrics."""
from __future__ import annotations
import json
import sqlite3
import threading
import time
from contextlib import contextmanager

from . import config

_lock = threading.Lock()
_SCHEMA = """CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    endpoint TEXT NOT NULL,
    engine TEXT,
    routed INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    latency_ms REAL,
    llm_ms REAL,
    llm_calls INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    output_tokens_per_s REAL,
    cost_usd REAL,
    abstained INTEGER,
    citation_coverage REAL,
    detail TEXT
)"""


@contextmanager
def _db():
    with _lock:
        conn = sqlite3.connect(config.METRICS_DB_PATH, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            with conn:  # commits on success
                conn.execute(_SCHEMA)
                yield conn
        finally:
            conn.close()


def record(endpoint, status, metrics=None, engine=None, routed=False, quality=None, detail=None):
    """Persist one request. Metrics failures must never fail the request itself."""
    metrics, quality = metrics or {}, quality or {}
    try:
        with _db() as conn:
            conn.execute(
                "INSERT INTO requests (ts, endpoint, engine, routed, status, latency_ms, llm_ms, llm_calls, prompt_tokens,"
                " completion_tokens, output_tokens_per_s, cost_usd, abstained, citation_coverage, detail)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (time.time(), endpoint, engine, int(routed), status, metrics.get("latency_ms"), metrics.get("llm_ms"),
                 metrics.get("llm_calls"), metrics.get("prompt_tokens"), metrics.get("completion_tokens"),
                 metrics.get("output_tokens_per_s"), metrics.get("cost_usd"),
                 None if "abstained" not in quality else int(quality["abstained"]), quality.get("citation_coverage"),
                 json.dumps({"stage_ms": metrics.get("stage_ms"), "ttft_ms": metrics.get("ttft_ms"), **(detail or {})})))
    except sqlite3.Error as error:
        print(f"  [metrics] failed to record {endpoint}: {error}", flush=True)


def _percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return round(values[lo] + (values[hi] - values[lo]) * (k - lo), 1)


def _latency(values):
    return {"p50": _percentile(values, .5), "p95": _percentile(values, .95), "p99": _percentile(values, .99),
            "mean": round(sum(values) / len(values), 1) if values else None, "max": round(max(values), 1) if values else None}


def _mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


def summary(window_minutes=60, endpoint=None):
    since = time.time() - window_minutes * 60
    query = "SELECT * FROM requests WHERE ts >= ?" + (" AND endpoint = ?" if endpoint else "") + " ORDER BY ts"
    with _db() as conn:
        rows = [dict(r) for r in conn.execute(query, (since, endpoint) if endpoint else (since,))]
    # Cache hits are neither errors nor model work: counted apart so they don't flatter latency.
    ok = [r for r in rows if r["status"] == "ok"]
    cached = [r for r in rows if r["status"] == "cached"]
    stage_ms, ttft_ms = {}, []
    for r in ok:
        detail = json.loads(r["detail"] or "{}")
        for stage, ms in (detail.get("stage_ms") or {}).items():
            stage_ms.setdefault(stage, []).append(ms)
        if detail.get("ttft_ms") is not None:
            ttft_ms.append(detail["ttft_ms"])
    # Throughput over the span actually observed, so a short burst isn't diluted by an idle window.
    span_s = max(rows[-1]["ts"] - rows[0]["ts"], 1.0) if len(rows) > 1 else window_minutes * 60
    llm_s = sum((r["llm_ms"] or 0) for r in ok) / 1000
    completion = sum((r["completion_tokens"] or 0) for r in ok)
    prompt = sum((r["prompt_tokens"] or 0) for r in ok)
    engines = {}
    for r in ok:
        if r["engine"]:
            e = engines.setdefault(r["engine"], {"requests": 0, "routed": 0, "latency_ms": []})
            e["requests"] += 1
            e["routed"] += r["routed"]
            e["latency_ms"].append(r["latency_ms"])
    return {
        "window_minutes": window_minutes,
        "requests": len(rows),
        "errors": len(rows) - len(ok) - len(cached),
        "error_rate": round((len(rows) - len(ok) - len(cached)) / len(rows), 4) if rows else None,
        "latency_ms": _latency([r["latency_ms"] for r in ok if r["latency_ms"] is not None]),
        "ttft_ms": _latency(ttft_ms),
        "cache": {"hits": len(cached), "hit_rate": round(len(cached) / (len(ok) + len(cached)), 4) if ok or cached else None,
                  "latency_ms": _latency([r["latency_ms"] for r in cached if r["latency_ms"] is not None])},
        "stage_latency_ms": {s: _latency(v) for s, v in stage_ms.items()},
        "throughput": {
            "requests_per_min": round(len(rows) / span_s * 60, 3) if rows else 0.0,
            "output_tokens_per_s": round(completion / llm_s, 2) if llm_s else None,
        },
        "tokens": {"prompt": prompt, "completion": completion, "total": prompt + completion,
                   "mean_per_request": round((prompt + completion) / len(ok), 1) if ok else None},
        "cost_usd": {"total": round(sum((r["cost_usd"] or 0) for r in ok), 6),
                     "mean_per_request": _mean([r["cost_usd"] for r in ok])},
        "quality": {"abstention_rate": _mean([r["abstained"] for r in ok]),
                    "mean_citation_coverage": _mean([r["citation_coverage"] for r in ok])},
        "engines": {e: {"requests": v["requests"], "routed_share": round(v["routed"] / v["requests"], 3),
                        "latency_ms": _latency(v["latency_ms"])} for e, v in engines.items()},
        "by_endpoint": {ep: sum(r["endpoint"] == ep for r in rows) for ep in sorted({r["endpoint"] for r in rows})},
    }


def recent(limit=50):
    with _db() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM requests ORDER BY ts DESC LIMIT ?", (limit,))]
    for r in rows:
        r["detail"] = json.loads(r["detail"] or "{}")
    return rows
