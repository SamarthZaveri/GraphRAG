"""Rate limits, private workspaces, public-mode admin gating and router compatibility."""
import time
from unittest.mock import patch

import pytest
from fastapi import HTTPException

WS = "ab" * 12


def _client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app import config, llm_client, main
    monkeypatch.setattr(config, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(llm_client, "is_available", lambda: True)
    return TestClient(main.app), main


def _fake_ingest(seen):
    def fake(paths, mode="replace", progress=None):
        from app import workspace
        from app.models import IngestResponse
        seen.update(paths=paths, workspace=workspace.current(), state_dir=workspace.state_dir())
        return IngestResponse(doc_ids=[p.stem for p in paths], num_chunks=1, num_entities=1, num_triples=0, num_communities=0)
    return fake


def test_rate_limit_returns_429_with_retry_after(tmp_path, monkeypatch):
    from app import guard
    client, _ = _client(tmp_path, monkeypatch)
    monkeypatch.setattr(guard, "limiter", guard.RateLimiter({"query": "2/60", "compare": "1/60", "ingest": "1/60", "default": "100/60"}))
    ok = client.get("/api/metrics")
    assert ok.headers["x-ratelimit-group"] == "default"
    with patch("app.main._traced_answer", side_effect=RuntimeError("boom")):
        codes = [client.post("/api/query", json={"question": "Q?", "engine": "vector_rag"}).status_code for _ in range(3)]
    assert codes[:2] == [500, 500] and codes[2] == 429
    blocked = client.post("/api/query", json={"question": "Q?", "engine": "vector_rag"})
    assert int(blocked.headers["retry-after"]) >= 1 and "Rate limit" in blocked.json()["detail"]
    assert client.get("/api/health").status_code == 200  # never limited


def test_workspace_upload_is_private_and_scoped(tmp_path, monkeypatch):
    from app import jobs, workspace
    client, main = _client(tmp_path, monkeypatch)
    seen = {}
    monkeypatch.setattr(main, "_ingest_documents", _fake_ingest(seen))
    job = client.post("/api/ingest/upload", headers={"X-Ledger-Workspace": WS},
                      files=[("files", ("filing.txt", b"Revenue rose 10%.", "text/plain"))]).json()
    jobs._executor.submit(lambda: None).result()
    assert seen["workspace"] == WS and WS in str(seen["state_dir"])
    assert seen["paths"][0].parent == workspace.root(WS) / "uploads"
    assert not (tmp_path / "uploads").exists()  # the shared corpus is untouched
    # Only the owner can see the job; the status endpoint reports the caller's workspace.
    assert client.get(f"/api/ingest/jobs/{job['id']}").status_code == 404
    assert client.get(f"/api/ingest/jobs/{job['id']}", headers={"X-Ledger-Workspace": WS}).json()["status"] == "done"
    status = client.get("/api/status", headers={"X-Ledger-Workspace": WS}).json()
    assert status["workspace"]["docs"] == ["filing"]
    assert client.delete("/api/workspace", headers={"X-Ledger-Workspace": WS}).json() == {"deleted": WS}
    assert not workspace.exists(WS)


def test_workspace_quota_rejects_oversized_text(tmp_path, monkeypatch):
    from app import config
    client, _ = _client(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "MAX_WORKSPACE_CHARS", 10)
    r = client.post("/api/ingest/upload", headers={"X-Ledger-Workspace": WS},
                    files=[("files", ("big.txt", b"x" * 50, "text/plain"))])
    assert r.status_code == 413 and "characters" in r.json()["detail"]


def test_queries_without_a_known_workspace_use_the_shared_corpus(tmp_path, monkeypatch):
    from app import models, workspace
    client, main = _client(tmp_path, monkeypatch)
    seen = []
    def answer(engine, question, mode="auto"):
        seen.append(workspace.current())
        return models.QueryResponse(question=question, mode_used="local", answer="ok", citations=[])
    with patch.object(main.engines, "answer_question", side_effect=answer):
        client.post("/api/query", json={"question": "Q?", "engine": "vector_rag"}, headers={"X-Ledger-Workspace": WS})
        workspace.touch(WS)
        client.post("/api/query", json={"question": "Q?", "engine": "vector_rag"}, headers={"X-Ledger-Workspace": WS})
        client.post("/api/query", json={"question": "Q?", "engine": "vector_rag"}, headers={"X-Ledger-Workspace": "../etc"})
    assert seen == [None, WS, None]


def test_public_mode_locks_shared_writes_and_debug_endpoints(tmp_path, monkeypatch):
    from app import config
    client, _ = _client(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "PUBLIC_MODE", True)
    monkeypatch.setattr(config, "ADMIN_TOKEN", "s3cret")
    assert client.post("/api/benchmark/run").status_code == 403
    assert client.get("/api/graph").status_code == 403
    assert client.post("/api/ingest/sample").status_code == 403
    assert client.post("/api/ingest/upload", files=[("files", ("a.txt", b"text", "text/plain"))]).status_code == 403
    assert client.get("/api/graph", headers={"X-Admin-Token": "s3cret"}).status_code == 200


def test_llm_slot_sheds_load_with_503(monkeypatch):
    import threading
    from app import config, guard
    monkeypatch.setattr(guard, "_llm_slots", threading.BoundedSemaphore(1))
    monkeypatch.setattr(config, "LLM_QUEUE_WAIT_SECONDS", 0.01)
    with guard.llm_slot():
        with pytest.raises(HTTPException) as busy:
            with guard.llm_slot():
                pass
    assert busy.value.status_code == 503 and busy.value.headers["Retry-After"] == "10"


def test_expired_workspaces_are_cleaned_up():
    from app import workspace
    workspace.touch(WS)
    meta = workspace.root(WS) / "meta.json"
    meta.write_text(meta.read_text().replace('"last_used": ', '"last_used": 1, "_": '))
    workspace.cleanup(force=True)
    assert not workspace.exists(WS)


@pytest.mark.parametrize("trained, serving, ok", [
    ("qwen2.5:14b", "Qwen/Qwen2.5-14B-Instruct", True),
    ("qwen2.5:14b", "qwen2.5:14b", True),
    ("qwen2.5:14b", "Qwen/Qwen2.5-7B-Instruct", False),
])
def test_router_policy_accepts_same_model_across_providers(trained, serving, ok, monkeypatch):
    from app import config, corpus_router
    from app.query_bandit import LinUCBBandit
    bandit = LinUCBBandit(8, ["graphrag", "vector_rag", "hybrid_rag"])
    bandit.metadata = {"retrieval_version": config.RETRIEVAL_VERSION, "answer_model": trained,
                       "extraction_model": trained, "llm_backend": "ollama"}
    monkeypatch.setattr(config, "ANSWER_MODEL", serving)
    monkeypatch.setattr(config, "EXTRACTION_MODEL", serving)
    assert corpus_router._policy_matches_config(bandit) is ok
