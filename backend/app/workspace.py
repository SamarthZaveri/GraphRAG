"""Private, expiring per-visitor corpora.

A workspace is a separate graph-state directory, vector collection and uploads
folder. The active workspace is bound to the request context (see the guard
middleware in main.py); with none bound, every path resolves to the shared
corpus exactly as before, so offline training and evaluation are unaffected.
The trained router policy stays global: it is a model, not corpus state."""
from __future__ import annotations
import contextvars
import json
import re
import shutil
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from . import config

_current = contextvars.ContextVar("ledger_workspace", default=None)
ID_PATTERN = re.compile(r"^[a-f0-9]{16,32}$")
SHARED_COLLECTION = "ledger_vector_baseline"
_lock = threading.Lock()
_last_cleanup = 0.0


def valid_id(ws_id) -> bool:
    return isinstance(ws_id, str) and bool(ID_PATTERN.match(ws_id))


def current() -> Optional[str]:
    return _current.get()


@contextmanager
def use(ws_id: Optional[str]):
    token = _current.set(ws_id)
    try:
        yield
    finally:
        _current.reset(token)


def root(ws_id: str) -> Path:
    return config.WORKSPACES_DIR / ws_id


def state_dir() -> Path:
    ws = current()
    path = config.GRAPH_STATE_DIR if ws is None else root(ws) / "graph_state"
    path.mkdir(parents=True, exist_ok=True)
    return path


def uploads_dir() -> Path:
    ws = current()
    return config.UPLOADS_DIR if ws is None else root(ws) / "uploads"


def collection_name(ws_id: Optional[str] = None) -> str:
    ws = ws_id if ws_id is not None else current()
    return SHARED_COLLECTION if ws is None else f"ws_{ws}"


def exists(ws_id: str) -> bool:
    return valid_id(ws_id) and (root(ws_id) / "meta.json").exists()


def _meta(ws_id):
    try:
        return json.loads((root(ws_id) / "meta.json").read_text())
    except (OSError, ValueError):
        return None


def touch(ws_id: str, **fields):
    """Create or refresh a workspace; its expiry is measured from the last use."""
    with _lock:
        root(ws_id).mkdir(parents=True, exist_ok=True)
        meta = _meta(ws_id) or {"created": time.time(), "chars": 0}
        meta.update(fields, last_used=time.time())
        (root(ws_id) / "meta.json").write_text(json.dumps(meta))
        return meta


def info(ws_id: str) -> Optional[dict]:
    if not exists(ws_id):
        return None
    meta = _meta(ws_id) or {}
    docs = sorted(p.stem for p in (root(ws_id) / "uploads").glob("*") if p.is_file())
    return {"id": ws_id, "docs": docs, "chars": meta.get("chars", 0),
            "expires_at": meta.get("last_used", time.time()) + config.WORKSPACE_TTL_HOURS * 3600,
            "limits": {"max_docs": config.MAX_WORKSPACE_DOCS, "max_chars": config.MAX_WORKSPACE_CHARS}}


def delete(ws_id: str):
    from . import vector_baseline
    vector_baseline.drop_collection(collection_name(ws_id))
    shutil.rmtree(root(ws_id), ignore_errors=True)


def cleanup(force=False):
    """Delete expired workspaces, then the least recently used beyond MAX_WORKSPACES. Throttled."""
    global _last_cleanup
    if not force and time.time() - _last_cleanup < 600:
        return
    _last_cleanup = time.time()
    if not config.WORKSPACES_DIR.exists():
        return
    alive = []
    for path in config.WORKSPACES_DIR.iterdir():
        meta = _meta(path.name) if valid_id(path.name) else None
        last = (meta or {}).get("last_used", 0)
        if time.time() - last > config.WORKSPACE_TTL_HOURS * 3600:
            delete(path.name)
        else:
            alive.append((last, path.name))
    excess = len(alive) - config.MAX_WORKSPACES
    for _, ws_id in sorted(alive)[:max(0, excess)]:
        delete(ws_id)
