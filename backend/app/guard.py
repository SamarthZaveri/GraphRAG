"""Request guardrails: per-IP rate limits, workspace binding, admin checks and an LLM concurrency cap."""
from __future__ import annotations
import hmac
import math
import threading
import time
from collections import deque
from contextlib import contextmanager

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from . import config, workspace

WORKSPACE_HEADER = "x-ledger-workspace"
ADMIN_HEADER = "x-admin-token"


def _parse(rule):
    count, seconds = rule.split("/")
    return int(count), float(seconds)


class RateLimiter:
    """Sliding-window counter per (client, group), in process memory."""

    def __init__(self, rules):
        self.rules = {group: _parse(rule) for group, rule in rules.items()}
        self.hits = {}
        self.lock = threading.Lock()
        self.last_prune = time.monotonic()

    def hit(self, client, group):
        """Record a request; returns (allowed, limit, remaining, retry_after_seconds)."""
        limit, window = self.rules[group]
        now = time.monotonic()
        with self.lock:
            q = self.hits.setdefault((client, group), deque())
            while q and now - q[0] >= window:
                q.popleft()
            if len(q) >= limit:
                return False, limit, 0, max(1, math.ceil(window - (now - q[0])))
            q.append(now)
            if now - self.last_prune > 300:
                self._prune(now)
            return True, limit, limit - len(q), 0

    def _prune(self, now):
        self.last_prune = now
        longest = max(window for _, window in self.rules.values())
        for key in [k for k, q in self.hits.items() if not q or now - q[-1] >= longest]:
            del self.hits[key]


limiter = RateLimiter(config.RATE_LIMITS)


def classify(path, method):
    if not path.startswith("/api/") or path == "/api/health":
        return None
    if path.startswith("/api/query/compare"):
        return "compare"
    if path.startswith("/api/query"):
        return "query"
    if path.startswith("/api/ingest/") and method == "POST":
        return "ingest"
    return "default"


def client_ip(scope, headers):
    if config.TRUST_PROXY and headers.get("x-forwarded-for"):
        return headers["x-forwarded-for"].split(",")[0].strip()
    client = scope.get("client")
    return client[0] if client else "unknown"


class GuardMiddleware:
    """Pure ASGI (not BaseHTTPMiddleware) so the workspace context reaches endpoint threads."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        path, method = scope.get("path", ""), scope.get("method", "GET")
        group = classify(path, method) if method != "OPTIONS" else None
        rate_headers = []
        if group:
            allowed, limit, remaining, retry = limiter.hit(client_ip(scope, headers), group)
            rate_headers = [(b"x-ratelimit-limit", str(limit).encode()), (b"x-ratelimit-remaining", str(remaining).encode()),
                            (b"x-ratelimit-group", group.encode())]
            if not allowed:
                window = int(limiter.rules[group][1])
                response = JSONResponse(
                    {"detail": f"Rate limit reached: {limit} {group} requests per {window}s. Try again in {retry}s.",
                     "retry_after": retry, "group": group},
                    status_code=429, headers={"Retry-After": str(retry), "X-RateLimit-Limit": str(limit),
                                              "X-RateLimit-Remaining": "0", "X-RateLimit-Group": group})
                return await response(scope, receive, send)
        ws = headers.get(WORKSPACE_HEADER)
        # Bind an existing workspace, or a new one only on upload (the one call that creates it).
        if not workspace.valid_id(ws) or not (workspace.exists(ws) or path == "/api/ingest/upload"):
            ws = None

        async def send_with_headers(message):
            if message["type"] == "http.response.start" and rate_headers:
                message.setdefault("headers", [])
                message["headers"] = list(message["headers"]) + rate_headers
            await send(message)

        with workspace.use(ws):
            await self.app(scope, receive, send_with_headers)


def is_admin(request: Request) -> bool:
    token = request.headers.get(ADMIN_HEADER, "")
    return bool(config.ADMIN_TOKEN) and hmac.compare_digest(token, config.ADMIN_TOKEN)


def require_admin_in_public_mode(request: Request):
    """Dependency for shared-corpus writes and expensive or debug endpoints."""
    if config.PUBLIC_MODE and not is_admin(request):
        raise HTTPException(403, "This action is disabled on the public deployment.")


_llm_slots = threading.BoundedSemaphore(max(1, config.MAX_CONCURRENT_LLM_REQUESTS))


@contextmanager
def llm_slot():
    """Admit a bounded number of LLM-backed requests; shed the rest fast with a retryable 503."""
    if not _llm_slots.acquire(timeout=config.LLM_QUEUE_WAIT_SECONDS):
        raise HTTPException(503, "The model is busy with other questions. Try again in a few seconds.",
                            headers={"Retry-After": "10"})
    try:
        yield
    finally:
        _llm_slots.release()
