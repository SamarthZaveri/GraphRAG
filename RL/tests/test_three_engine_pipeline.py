"""Offline tests for retrieval, API transport, balanced QA and reward handling."""
import json
import sys
from pathlib import Path
from collections import Counter
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
import pytest



def test_bm25_preserves_periods_and_exact_metric_names():
    from app.lexical import bm25_rank
    docs = ["Acme Q2 2025 adjusted EBITDA was 42.5 million", "Acme Q2 2024 revenue was 10 million", "unrelated text"]
    assert bm25_rank("2025 EBITDA", docs)[0][0] == 0
    assert bm25_rank("doesnotexist", docs) == []
    assert bm25_rank("EBITDA", []) == []


def test_rrf_rewards_agreement_and_deduplicates():
    from app.lexical import reciprocal_rank_fusion
    assert reciprocal_rank_fusion([["a", "b"], ["b", "c"]])[0] == "b"
    assert reciprocal_rank_fusion([["a", "a"], ["b"]]) == ["a", "b"]


def test_hybrid_uses_raw_collection_and_both_rankings():
    from app import hybrid_search
    collection = Mock()
    collection.count.return_value = 3
    collection.get.return_value = {"ids": ["a", "b", "c"],
        "documents": ["Acme revenue", "Acme EBITDA 2025", "other company"],
        "metadatas": [{"doc_id": "acme"}, {"doc_id": "acme"}, {"doc_id": "other"}]}
    collection.query.return_value = {"ids": [["c", "b", "a"]]}
    with patch.object(hybrid_search.vector_baseline, "get_collection", return_value=collection):
        chunks = hybrid_search.retrieve("2025 EBITDA", 2)
    assert [c["chunk_id"] for c in chunks] == ["b", "c"]
    collection.query.assert_called_once()


def test_graph_context_excludes_unrelated_raw_chunks():
    from app import query_engine
    from app.graph_store import GraphStore
    store = GraphStore()
    for name in ("A", "B", "X", "Y"):
        store.graph.add_node(name, type="Metric", description=name)
    store.graph.add_edge("A", "B", predicate="reports", doc_id="report", chunk_id="c1", evidence="Revenue 10")
    store.graph.add_edge("X", "Y", predicate="reports", doc_id="other", chunk_id="c2")
    store.chunks = {"c1": {"doc_id": "report", "text": "Revenue 10"}, "c2": {"doc_id": "other", "text": "UNRELATED"}}
    with patch.object(query_engine, "_fuzzy_match_nodes", return_value=["A"]), patch.object(query_engine.rgcn, "nearest_nodes", return_value=[]):
        context = query_engine._local_context("Revenue?", store, ["A"])
    assert context[-1] == ["[report | c1]\nRevenue 10"]
    assert "vector_baseline" not in Path(query_engine.__file__).read_text(encoding="utf-8")


def test_generation_balances_categories_and_checks_quotes():
    import generate_benchmark as generator
    sources = ["Acme reported revenue of 100 million.", "Beta reported revenue of 80 million."]
    evidence = [{"doc_id": "a", "quote": sources[0]}, {"doc_id": "b", "quote": sources[1]}]
    def fake_chat(model, system, user, **kwargs):
        if system.startswith("Audit"):
            return {"checks": [{"index": i, "valid": True, "reason": "supported"} for i, _ in enumerate(json.loads(user))]}
        category = next(c for c in generator.CATEGORIES if f"category '{c}'" in system)
        return {"questions": [{"question": f"{category} question {i}?", "category": category,
            "reference_answer": "Grounded answer", "reasoning": "Compare the supported facts", "evidence": evidence} for i in range(3)]}
    with patch.object(generator, "chat_json", side_effect=fake_chat):
        questions = generator.generate_benchmark_for_corpus(sources, ["a", "b"], per_category=3)
    assert Counter(q["category"] for q in questions) == {c: 3 for c in generator.CATEGORIES}
    assert len({q["id"] for q in questions}) == 12
    bad = {**questions[0], "evidence": [{"doc_id": "a", "quote": "This sentence is invented."}]}
    assert generator.clean_question(bad, "local", dict(zip(["a", "b"], sources)), set()) is None


def test_generation_does_not_silently_accept_shortfalls():
    import generate_benchmark as generator
    with patch.object(generator, "chat_json", return_value={"questions": []}):
        with pytest.raises(ValueError, match="incomplete"):
            generator.generate_benchmark_for_corpus(["A sufficiently long source document."], ["a"], per_category=1)


def test_judge_is_blinded_and_maps_scores_to_engines():
    from app import benchmark
    answers = {"graphrag": "answer graph", "vector_rag": "answer vector", "hybrid_rag": "answer hybrid"}
    expected = {"answer graph": 1, "answer vector": 3, "answer hybrid": 5}
    def fake_chat(model, system, user, **kwargs):
        data = json.loads(user)
        assert set(data["candidates"]) == {"A", "B", "C"}
        assert not any(engine in user for engine in answers)
        return {"scores": {label: {"score": expected[answer], "rationale": "Checked evidence"} for label, answer in data["candidates"].items()}}
    with patch.object(benchmark, "chat_json", side_effect=fake_chat):
        scores = benchmark.judge_answers("Question?", "Reference", answers)
    assert {engine: data["score"] for engine, data in scores.items()} == {"graphrag": 1, "vector_rag": 3, "hybrid_rag": 5}


def test_judge_failure_never_becomes_zero_reward():
    from app import benchmark
    with patch.object(benchmark, "chat_json", return_value={"scores": {}}) as call:
        with pytest.raises(benchmark.JudgeError):
            benchmark.judge_answers("Question?", "Reference", {"graphrag": "A", "vector_rag": "B", "hybrid_rag": "C"})
    assert call.call_count == 3


def test_api_transport_sends_hosted_model_and_auth_without_ollama_options():
    from app import config, llm_client
    response = Mock(ok=True, status_code=200)
    response.json.return_value = {"choices": [{"finish_reason": "stop", "message": {"content": '{"ok": true}'}}]}
    with patch.object(config, "LLM_BACKEND", "api"), patch.object(config, "API_KEY", "test-token"), patch.object(llm_client.requests, "post", return_value=response) as post:
        assert llm_client.chat_json("hosted-model", "Return JSON", "Question") == {"ok": True}
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer test-token"
    payload = post.call_args.kwargs["json"]
    assert payload["model"] == "hosted-model"
    assert "options" not in payload


def test_api_retries_rate_limit_and_rejects_truncated_output():
    from app import config, llm_client
    limited = Mock(ok=False, status_code=429)
    success = Mock(ok=True, status_code=200)
    success.json.return_value = {"choices": [{"finish_reason": "length", "message": {"content": "partial"}}]}
    with patch.object(config, "LLM_BACKEND", "api"), patch.object(config, "API_KEY", "test"), patch.object(llm_client.requests, "post", side_effect=[limited, success]), patch.object(llm_client.time, "sleep"):
        with pytest.raises(llm_client.LLMError, match="truncated"):
            llm_client.chat("model", "system", "user")


def test_three_arm_training_and_legacy_reward_rejection(tmp_path):
    import train_shared_router as trainer
    from app import config
    from app.router_features import build_query_context
    context = build_query_context(np.array([0.2, 0.2, 0.2, 1]), "local").tolist()
    rows = [{"corpus": "c", "question": f"q{i}", "category": "local", "context": context,
             "reward_graphrag": 0.1, "reward_vector_rag": 0.4, "reward_hybrid_rag": 0.95, "sample_weight": 1,
             "metadata": {"retrieval_version": config.RETRIEVAL_VERSION, "answer_model": "m", "judge_model": "j", "extraction_model": "e", "llm_backend": "api"}} for i in range(80)]
    policy = trainer.train(rows, 1.0, 0)
    assert max(policy.arms, key=lambda arm: policy.predicted_reward(arm, context)) == "hybrid_rag"
    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text(json.dumps({k: v for k, v in rows[0].items() if k != "metadata"}))
    with pytest.raises(ValueError, match="Historical"):
        trainer.load_results(legacy)


def test_api_selects_hybrid_and_comparison_includes_all_three():
    from app import main, models
    def response(engine, question, mode="auto"):
        return models.QueryResponse(question=question, mode_used="local", answer=engine, citations=[])
    with patch.object(main.engines, "answer_question", side_effect=response):
        result = main.query(models.QueryRequest(question="Q?", engine="hybrid_rag"))
    assert result.engine_used == "hybrid_rag"
    with patch.object(main.query_engine, "answer_question", return_value=response("graph", "Q?")), patch.object(main.vector_baseline, "answer_question", return_value=response("vector", "Q?")), patch.object(main.engines, "answer_question", side_effect=response):
        assert main.query_compare(models.QueryRequest(question="Q?")).hybrid_rag.answer == "hybrid_rag"


def test_nvidia_nemotron_request_controls_thinking():
    from app import config, llm_client
    response = Mock(ok=True, status_code=200)
    response.json.return_value = {"choices": [{"finish_reason": "stop", "message": {"content": '{"ok": true}'}}]}
    with patch.object(config, "LLM_BACKEND", "api"), patch.object(config, "API_KEY", "test-token"), patch.object(config, "API_BASE_URL", "https://integrate.api.nvidia.com/v1"), patch.object(config, "NVIDIA_THINKING", False), patch.object(llm_client.requests, "post", return_value=response) as post:
        assert llm_client.chat_json("nvidia/nemotron-3-super-120b-a12b", "Return JSON", "Question") == {"ok": True}
    assert post.call_args.kwargs["json"]["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.parametrize("error, expected", [
    ("ConnectTimeout", "Connection timed out"),
    ("ReadTimeout", "Response timed out"),
    ("SSLError", "TLS handshake"),
    ("ProxyError", "Proxy connection failed"),
    ("ConnectionError", "DNS lookup failed"),
])
def test_transport_errors_are_actionable_and_do_not_expose_credentials(error, expected):
    from app import config, llm_client
    failure = getattr(llm_client.requests.exceptions, error)("getaddrinfo failed; https://user:secret@proxy.invalid")
    with patch.object(config, "LLM_BACKEND", "api"), patch.object(config, "API_KEY", "test-token"), patch.object(config, "LLM_RETRIES", 1), patch.object(llm_client.requests, "post", side_effect=failure):
        with pytest.raises(llm_client.LLMError) as caught:
            llm_client.chat("model", "system", "user")
    assert expected in str(caught.value)
    assert "secret" not in str(caught.value)
    assert "test-token" not in str(caught.value)


def test_network_probe_does_not_send_api_key(capsys):
    import importlib.util
    from app import config
    spec = importlib.util.spec_from_file_location("check_models", Path(__file__).resolve().parents[2] / "scripts" / "check_models.py")
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    with patch.object(config, "LLM_BACKEND", "api"), patch.object(config, "API_KEY", "test-token"), patch.object(checker.requests, "get", return_value=Mock(status_code=401)) as get:
        checker.check_network()
    assert "headers" not in get.call_args.kwargs
    assert get.call_args.kwargs["timeout"] == (10, 20)
    assert "HTTPS works" in capsys.readouterr().out



def test_evidence_allows_only_whitespace_changes_and_saves_original():
    import generate_benchmark as generator
    text = "Revenue was\n 100 million in Q2\t2026."
    q = {"question": "Revenue?", "category": "local", "reference_answer": "100 million", "reasoning": "Quoted metric", "evidence": [{"doc_id": "a", "quote": "Revenue was 100 million in Q2 2026."}]}
    cleaned = generator.clean_question(q, "local", {"a": text}, set())
    assert cleaned["evidence"][0]["quote"] == text
    q["evidence"][0]["quote"] = "Revenue was 101 million in Q2 2026."
    rejected = []
    assert generator.clean_question(q, "local", {"a": text}, set(), rejected) is None
    assert "Quotation not found" in rejected[0]


def test_audit_rejections_feed_back_into_next_generation_attempt():
    import generate_benchmark as generator
    source = "Acme revenue was 100 million and profit was 20 million."
    calls = Counter()
    saw_feedback = []
    def chat(model, system, user, **kwargs):
        if system.startswith("Audit"):
            category = json.loads(user)[0]["category"]
            valid = calls[category] > 1
            return {"checks": [{"index": 0, "valid": valid, "reason": "Use a supported unit"}]}
        category = next(c for c in generator.CATEGORIES if f"category '{c}'" in system)
        calls[category] += 1
        if calls[category] > 1:
            saw_feedback.append("Use a supported unit" in user)
        return {"questions": [{"question": category + " question?", "category": category, "reference_answer": "100", "reasoning": "Evidence", "evidence": [{"doc_id": "a", "quote": "Acme revenue was 100 million"}, {"doc_id": "a", "quote": "profit was 20 million."}]}]}
    with patch.object(generator, "chat_json", side_effect=chat):
        questions = generator.generate_benchmark_for_corpus([source], ["a"], per_category=1)
    assert len(questions) == 4
    assert len(saw_feedback) == 4 and all(saw_feedback)





def test_extraction_retries_truncation_with_larger_budget(isolated_extraction_cache):
    from app import config, extraction, llm_client
    chunk = {"doc_id": "a", "chunk_id": "a::chunk0", "text": "Revenue was 10 million."}
    valid = {"entities": [{"name": "Acme", "type": "Company"}], "triples": []}
    with patch.object(config, "EXTRACTION_MAX_TOKENS", 4096), patch.object(config, "EXTRACTION_TOKEN_LIMIT", 16384), patch.object(extraction, "chat_json", side_effect=[llm_client.OutputTruncatedError("truncated"), valid]) as chat:
        result = extraction.extract_from_chunk(chunk)
    assert [call.kwargs["max_tokens"] for call in chat.call_args_list] == [4096, 8192]
    assert result.entities[0].name == "Acme"


def test_extraction_truncation_retries_are_bounded(isolated_extraction_cache):
    from app import config, extraction, llm_client
    chunk = {"doc_id": "a", "chunk_id": "a::chunk0", "text": "Revenue was 10 million."}
    with patch.object(config, "EXTRACTION_MAX_TOKENS", 4096), patch.object(config, "EXTRACTION_TOKEN_LIMIT", 10000), patch.object(extraction, "chat_json", side_effect=llm_client.OutputTruncatedError("truncated")) as chat:
        with pytest.raises(llm_client.OutputTruncatedError, match="a::chunk0 at 10000"):
            extraction.extract_from_chunk(chunk)
    assert [call.kwargs["max_tokens"] for call in chat.call_args_list] == [4096, 8192, 10000]


def test_extraction_does_not_retry_other_llm_errors_with_more_tokens(isolated_extraction_cache):
    from app import extraction, llm_client
    chunk = {"doc_id": "a", "chunk_id": "a::chunk0", "text": "Revenue was 10 million."}
    with patch.object(extraction, "chat_json", side_effect=llm_client.LLMError("HTTP 401")) as chat:
        with pytest.raises(llm_client.LLMError, match="HTTP 401"):
            extraction.extract_from_chunk(chunk)
    assert chat.call_count == 1


@pytest.fixture
def isolated_extraction_cache(tmp_path, monkeypatch):
    from app import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    return tmp_path


def test_rate_limit_honors_retry_after_and_eventually_succeeds():
    from app import config, llm_client
    limited = Mock(ok=False, status_code=429, headers={"Retry-After": "45"})
    success = Mock(ok=True, status_code=200)
    success.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
    with patch.object(config, "LLM_BACKEND", "api"), patch.object(config, "API_KEY", "test"), patch.object(llm_client.requests, "post", side_effect=[limited, success]), patch.object(llm_client.time, "sleep") as sleep:
        assert llm_client.chat("m", "s", "u") == "ok"
    sleep.assert_called_once_with(45)


def test_persistent_rate_limit_has_bounded_retries():
    from app import config, llm_client
    limited = Mock(ok=False, status_code=429, headers={})
    with patch.object(config, "LLM_BACKEND", "api"), patch.object(config, "API_KEY", "test"), patch.object(config, "RATE_LIMIT_ATTEMPTS", 3), patch.object(llm_client.requests, "post", return_value=limited) as post, patch.object(llm_client.time, "sleep") as sleep:
        with pytest.raises(llm_client.LLMError, match="quota persists"):
            llm_client.chat("m", "s", "u")
    assert post.call_count == 3
    assert [call.args[0] for call in sleep.call_args_list] == [15, 30]


def test_retry_after_http_date():
    from app import llm_client
    response = Mock(headers={"Retry-After": "Thu, 01 Jan 1970 00:02:00 GMT"})
    with patch.object(llm_client.time, "time", return_value=100):
        assert llm_client.rate_limit_delay(response, 0) == 20


def test_checkpoint_reuses_success_and_invalidates_changed_source(isolated_extraction_cache):
    from app import extraction
    chunk = {"doc_id": "a", "chunk_id": "a::chunk0", "text": "Revenue was 10 million."}
    valid = {"entities": [{"name": "Acme", "type": "Company"}], "triples": []}
    with patch.object(extraction, "chat_json", return_value=valid) as chat:
        first = extraction.extract_from_chunk(chunk)
        assert extraction.extract_from_chunk(chunk) == first
        assert chat.call_count == 1
        extraction.extract_from_chunk({**chunk, "text": "Revenue was 11 million."})
        assert chat.call_count == 2
    assert len(list((isolated_extraction_cache / "extraction_cache").glob("*.json"))) == 2


def test_failed_extraction_creates_no_checkpoint(isolated_extraction_cache):
    from app import extraction, llm_client
    chunk = {"doc_id": "a", "chunk_id": "a::chunk0", "text": "Revenue was 10 million."}
    with patch.object(extraction, "chat_json", return_value={}):
        with pytest.raises(llm_client.LLMError, match="Invalid extraction JSON"):
            extraction.extract_from_chunk(chunk)
    assert not (isolated_extraction_cache / "extraction_cache").exists()
