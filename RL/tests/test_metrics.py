"""Request telemetry, metrics persistence and the metrics/RL endpoints."""
import json
from unittest.mock import Mock, patch


def test_ollama_usage_is_recorded_in_active_trace_only():
    from app import config, llm_client, telemetry
    response = Mock(ok=True, status_code=200)
    response.json.return_value = {"message": {"content": "answer"}, "done_reason": "stop",
                                  "prompt_eval_count": 120, "eval_count": 30, "eval_duration": 1_500_000_000}
    with patch.object(config, "LLM_BACKEND", "ollama"), patch.object(llm_client.requests, "post", return_value=response):
        llm_client.chat("m", "sys", "user")  # no trace: must be a no-op
        with telemetry.trace() as t, telemetry.span("generation"):
            llm_client.chat("m", "sys", "user")
    summary = t.summary()
    assert summary["prompt_tokens"] == 120 and summary["completion_tokens"] == 30
    assert summary["llm_calls"] == 1 and not summary["tokens_estimated"]
    assert summary["output_tokens_per_s"] == 20.0
    assert summary["llm_by_stage"]["generation"]["calls"] == 1


def test_missing_api_usage_falls_back_to_estimate_and_token_pricing():
    from app import config, llm_client, telemetry
    response = Mock(ok=True, status_code=200)
    response.json.return_value = {"choices": [{"finish_reason": "stop", "message": {"content": "x" * 400}}]}
    with patch.object(config, "LLM_BACKEND", "api"), patch.object(config, "API_KEY", "k"), \
            patch.object(config, "PRICE_INPUT_PER_M", 1.0), patch.object(config, "PRICE_OUTPUT_PER_M", 2.0), \
            patch.object(llm_client.requests, "post", return_value=response):
        with telemetry.trace() as t:
            llm_client.chat("m", "s" * 400, "")
        summary = t.summary()
    assert summary["tokens_estimated"]
    assert summary["cost_usd"] == round((100 * 1.0 + 100 * 2.0) / 1e6, 6)


def test_threads_report_into_the_callers_trace():
    from concurrent.futures import ThreadPoolExecutor
    from app import telemetry
    with telemetry.trace() as t:
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(telemetry.bind(lambda i: telemetry.record_llm_call("m", 0.1, 10, 5)), range(4)))
    assert t.summary()["llm_calls"] == 4


def test_query_attaches_metrics_and_persists_summary():
    from app import main, models, metrics_store
    def response(engine, question, mode="auto"):
        from app import telemetry
        telemetry.record_llm_call("m", 0.5, 100, 20)
        return models.QueryResponse(question=question, mode_used="local", answer="Revenue rose [doc_a].",
                                    citations=[models.Citation(doc_id="doc_a", chunk_id="c1", snippet="s"),
                                               models.Citation(doc_id="doc_b", chunk_id="c2", snippet="s")])
    with patch.object(main.engines, "answer_question", side_effect=response):
        result = main.query(models.QueryRequest(question="Q?", engine="vector_rag"))
        main.query_compare(models.QueryRequest(question="Q?"))
    assert result.metrics["total_tokens"] == 120
    assert result.quality["citation_coverage"] == 0.5
    summary = metrics_store.summary(60)
    assert summary["requests"] == 4 and summary["errors"] == 0
    assert summary["tokens"]["total"] == 480
    assert set(summary["engines"]) == {"graphrag", "vector_rag", "hybrid_rag"}
    assert summary["latency_ms"]["p95"] is not None


def test_rl_results_endpoint_reads_run_directory(tmp_path):
    from app import config, main
    (tmp_path / "dev").mkdir()
    (tmp_path / "dev" / "summary.json").write_text(json.dumps({"completed": 56}))
    (tmp_path / "policy.json").write_text(json.dumps({"arms": ["a"], "alpha": 1.0, "context_dim": 8, "metadata": {"split": "train"}}))
    with patch.object(config, "RL_RESULTS_DIR", tmp_path):
        data = main.get_rl_results()
    assert data["splits"]["dev"]["summary"] == {"completed": 56}
    assert data["splits"]["test"]["summary"] is None
    assert data["policy"]["metadata"] == {"split": "train"}


def test_analysis_reports_router_against_baselines():
    from analyze_results import analyze
    def rec(qid, g, v, h, pick):
        rewards = {"graphrag": g, "vector_rag": v, "hybrid_rag": h}
        return {"question_id": qid, "category": "local", "task": "text", "router_engine": pick,
                "router_reward": rewards[pick], **{"reward_" + e: s for e, s in rewards.items()},
                "answers": {e: {"answer": "ok", "citations": [{"doc_id": "d1"}]} for e in rewards}}
    records = [rec("q1", 1.0, 0.0, 0.5, "graphrag"), rec("q2", 0.0, 1.0, 0.5, "vector_rag")]
    result = analyze(records, {"q1": {"d1"}, "q2": {"d2"}})
    assert result["policies"]["router"]["mean"] == 1.0 == result["policies"]["oracle"]["mean"]
    assert result["router"]["optimal_pick_rate"] == 1.0
    assert result["retrieval"]["graphrag"]["doc_hit_rate"] == 0.5


def _client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app import config, llm_client, main
    monkeypatch.setattr(config, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(llm_client, "is_available", lambda: True)
    return TestClient(main.app), main


def test_upload_guardrails_reject_bad_batches_without_writing(tmp_path, monkeypatch):
    client, main = _client(tmp_path, monkeypatch)
    too_many = [("files", (f"d{i}.txt", b"text", "text/plain")) for i in range(6)]
    assert client.post("/api/ingest/upload", files=too_many).status_code == 413
    assert client.post("/api/ingest/upload", files=[("files", ("x.exe", b"MZ", "application/octet-stream"))]).status_code == 415
    assert client.post("/api/ingest/upload", files=[("files", ("fake.pdf", b"not a pdf", "application/pdf"))]).status_code == 415
    big = b"a" * (int(main.config.MAX_UPLOAD_MB * 1024 * 1024) + 1)
    assert client.post("/api/ingest/upload", files=[("files", ("ok.txt", b"fine", "text/plain")), ("files", ("big.txt", big, "text/plain"))]).status_code == 413
    assert not (tmp_path / "uploads").exists() or not any((tmp_path / "uploads").iterdir())


def test_upload_queues_job_with_sanitized_name(tmp_path, monkeypatch):
    client, main = _client(tmp_path, monkeypatch)
    seen = {}
    def fake_ingest(paths, mode="replace", progress=None):
        seen["paths"] = paths
        progress(1, paths[0].stem)
        from app.models import IngestResponse
        return IngestResponse(doc_ids=[p.stem for p in paths], num_chunks=1, num_entities=1, num_triples=0, num_communities=0)
    monkeypatch.setattr(main, "_ingest_documents", fake_ingest)
    job = client.post("/api/ingest/upload", files=[("files", ("../../evil name.txt", b"Revenue rose.", "text/plain"))]).json()
    main.jobs._executor.submit(lambda: None).result()  # wait for the single worker to drain
    done = client.get(f"/api/ingest/jobs/{job['id']}").json()
    assert done["status"] == "done", done
    assert seen["paths"][0].parent == tmp_path / "uploads" and seen["paths"][0].name == "evil_name.txt"
    assert done["result"]["metrics"]["per_document"] is not None
