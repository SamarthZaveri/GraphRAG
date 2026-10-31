"""Streamed answers (SSE, time to first token), the response cache and the PDF page limit."""
import json
from io import BytesIO
from unittest.mock import Mock, patch

import pytest


def _ollama_stream(pieces, done_reason="stop"):
    lines = [json.dumps({"message": {"content": p}, "done": False}) for p in pieces]
    lines.append(json.dumps({"message": {"content": ""}, "done": True, "done_reason": done_reason,
                             "prompt_eval_count": 50, "eval_count": len(pieces), "eval_duration": 1_000_000_000}))
    response = Mock(ok=True, status_code=200)
    response.iter_lines.return_value = lines
    response.__enter__ = lambda self: self
    response.__exit__ = lambda self, *a: None
    return response


def test_ollama_stream_delivers_pieces_and_records_ttft():
    from app import config, llm_client, telemetry
    got = []
    with patch.object(config, "LLM_BACKEND", "ollama"), \
            patch.object(llm_client.requests, "post", return_value=_ollama_stream(["Rev", "enue ", "rose."])) as post:
        with telemetry.trace() as t:
            text = llm_client.chat_stream("m", "sys", "user", got.append)
    assert text == "Revenue rose." and got == ["Rev", "enue ", "rose."]
    assert post.call_args.kwargs["json"]["stream"] is True
    summary = t.summary()
    assert summary["ttft_ms"] is not None and summary["completion_tokens"] == 3


def test_api_stream_parses_sse_and_flags_truncation():
    from app import config, llm_client
    events = ["data: " + json.dumps({"choices": [{"delta": {"content": "12.5"}}]}),
              "data: " + json.dumps({"choices": [{"delta": {"content": "%"}, "finish_reason": "length"}]}),
              "data: [DONE]"]
    response = _ollama_stream([])
    response.iter_lines.return_value = events
    with patch.object(config, "LLM_BACKEND", "api"), patch.object(config, "API_KEY", "k"), \
            patch.object(llm_client.requests, "post", return_value=response):
        with pytest.raises(llm_client.OutputTruncatedError):
            llm_client.chat_stream("m", "s", "u", lambda p: None)


def test_generate_answer_streams_only_inside_a_sink_and_resets_on_retry():
    from app import answering
    calls = []
    def stream(model, system, user, sink, max_tokens, temperature):
        calls.append(max_tokens)
        sink("partial")
        if len(calls) == 1:
            raise answering.OutputTruncatedError("loop")
        return "full answer"
    pieces = []
    with patch.object(answering, "chat", return_value="plain") as chat, patch.object(answering, "chat_stream", side_effect=stream):
        assert answering.generate_answer("q", "ctx") == "plain"  # no sink: the offline path
        with answering.streaming(pieces.append):
            assert answering.generate_answer("q", "ctx") == "full answer"
    assert chat.call_count == 1 and calls == [answering.config.ANSWER_MAX_TOKENS, answering.config.ANSWER_MAX_TOKENS * 2]
    assert pieces == ["partial", None, "partial"]


def _fake_engine(counter):
    def answer(engine, question, mode="auto", classification=None):
        from app import answering, models
        counter.append(engine)
        text = answering.generate_answer(question, "evidence")
        return models.QueryResponse(question=question, mode_used="local", answer=text,
                                    citations=[models.Citation(doc_id="doc_a", chunk_id="c1", snippet="s")])
    return answer


def _sse_events(body):
    out = []
    for block in body.split("\n\n"):
        lines = [l for l in block.splitlines() if not l.startswith(":")]
        if lines:
            out.append((lines[0][len("event: "):], json.loads(lines[1][len("data: "):])))
    return out


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """A shared corpus with graph-state files, so answers are cacheable."""
    from app import config, corpus_router
    state = tmp_path / "graph_state"
    state.mkdir()
    (state / "meta.json").write_text('{"docs": ["doc_a"]}')
    monkeypatch.setattr(config, "GRAPH_STATE_DIR", state)
    monkeypatch.setattr(corpus_router, "QUERY_BANDIT_PATH", state / "query_bandit.json")
    monkeypatch.setattr(config, "RESPONSE_CACHE", True)
    return state


def test_stream_endpoint_sends_meta_tokens_done_then_serves_cache(corpus):
    from fastapi.testclient import TestClient
    from app import main, answering, metrics_store
    calls = []
    def stream(model, system, user, sink, max_tokens, temperature):
        from app import telemetry
        telemetry.mark_first_token()
        for piece in ("Revenue ", "rose [doc_a]."):
            sink(piece)
        return "Revenue rose [doc_a]."
    client = TestClient(main.app)
    body = {"question": "How did revenue change?", "engine": "hybrid_rag"}
    with patch.object(main.engines, "answer_question", side_effect=_fake_engine(calls)), \
            patch.object(answering, "chat_stream", side_effect=stream):
        first = client.post("/api/query/stream", json=body)
        second = client.post("/api/query/stream", json={**body, "question": "  how did REVENUE change? "})
    events = _sse_events(first.text)
    assert first.headers["content-type"].startswith("text/event-stream")
    assert [e for e, _ in events] == ["meta", "token", "token", "done"]
    assert events[0][1]["engine"] == "hybrid_rag"
    done = events[-1][1]
    assert done["answer"] == "Revenue rose [doc_a]." and done["metrics"]["ttft_ms"] is not None and not done["cache_hit"]
    cached = _sse_events(second.text)
    assert [e for e, _ in cached] == ["done"] and cached[0][1]["cache_hit"] and cached[0][1]["answer"] == done["answer"]
    assert calls == ["hybrid_rag"]  # the second, normalized question never reached an engine
    summary = metrics_store.summary(60)
    assert summary["cache"]["hits"] == 1 and summary["errors"] == 0 and summary["ttft_ms"]["p50"] is not None


def test_stream_reports_errors_as_events(corpus):
    from fastapi.testclient import TestClient
    from app import main
    with patch.object(main.engines, "answer_question", side_effect=RuntimeError("model down")):
        response = TestClient(main.app).post("/api/query/stream", json={"question": "Q?", "engine": "vector_rag"})
    assert _sse_events(response.text)[-1] == ("error", {"detail": "model down"})


def test_cache_is_invalidated_by_reingestion_and_skips_truncated_answers(corpus):
    from app import main, models, answering
    calls = []
    with patch.object(main.engines, "answer_question", side_effect=_fake_engine(calls)), \
            patch.object(answering, "chat", return_value="Answer one."):
        req = models.QueryRequest(question="Q?", engine="vector_rag")
        assert not main.query(req).cache_hit
        assert main.query(req).cache_hit
        (corpus / "meta.json").write_text('{"docs": ["doc_a", "doc_b"]}')  # a new ingestion
        assert not main.query(req).cache_hit
        compare = models.QueryRequest(question="Q?")
        main.query_compare(compare)
        assert main.query_compare(compare).cache_hit
    assert calls.count("vector_rag") == 3
    with patch.object(main.engines, "answer_question", side_effect=_fake_engine(calls)), \
            patch.object(answering, "chat", return_value=answering.TRUNCATED_ANSWER):
        req = models.QueryRequest(question="Loops?", engine="vector_rag")
        main.query(req)
        assert not main.query(req).cache_hit


def test_empty_corpus_is_never_cached(tmp_path, monkeypatch):
    from app import config, response_cache
    monkeypatch.setattr(config, "RESPONSE_CACHE", True)
    monkeypatch.setattr(config, "GRAPH_STATE_DIR", tmp_path / "empty")
    assert response_cache.key("Q?", "auto", "auto") is None


def _pdf(pages):
    from pypdf import PdfWriter
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    out = BytesIO()
    writer.write(out)
    return out.getvalue()


def test_pdf_page_limit_rejects_long_documents(monkeypatch):
    from fastapi import HTTPException
    from app import config, main
    monkeypatch.setattr(config, "MAX_PDF_PAGES", 3)
    with pytest.raises(HTTPException) as error:
        main._validate_upload("annual_report.pdf", _pdf(4))
    assert error.value.status_code == 413 and "4 pages" in error.value.detail
    with pytest.raises(HTTPException) as error:  # within the limit, rejected only for having no text
        main._validate_upload("short.pdf", _pdf(2))
    assert error.value.status_code == 422


def test_suggested_questions_come_from_the_warm_up_file(tmp_path, monkeypatch):
    from app import config, main
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    assert main.get_suggested_questions() == []
    (tmp_path / "suggested_questions.json").write_text(json.dumps([{"question": "Q?", "category": "local"}]))
    assert main.get_suggested_questions()[0]["question"] == "Q?"


def test_warm_up_picks_every_category_per_corpus():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import warm_cache
    bundle = lambda name: {"id": name, "questions": [{"question": f"{name} {c} {i}", "category": c}
                                                     for c in ("local", "global", "multi_hop", "conflict") for i in range(3)]}
    manifest = {"splits": {"train": {"bundles": [bundle("a"), bundle("b"), bundle("c")]}}}
    picked = warm_cache.pick_questions(manifest, {"a", "b"}, 5)
    assert len(picked) == 10 and {q["corpus"] for q in picked} == {"a", "b"}
    assert {q["category"] for q in picked if q["corpus"] == "a"} == {"local", "global", "multi_hop", "conflict"}
    with pytest.raises(SystemExit):
        warm_cache.pick_questions(manifest, {"zzz"}, 5)
