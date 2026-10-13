"""Offline generation checkpoint, evidence and weighted update checks."""
import json
from unittest.mock import patch
import numpy as np
import pytest
from app import config
from app.query_bandit import LinUCBBandit

def test_weighted_bandit_updates_scale_covariance_and_reward():
    policy = LinUCBBandit(2, ["a"])
    before = policy.A["a"].copy()
    policy.update("a", [1, 2], .5, weight=2)
    assert np.allclose(policy.A["a"] - before, 2 * np.outer([1, 2], [1, 2]))
    assert np.allclose(policy.b["a"], [1, 2])
    with pytest.raises(ValueError, match="positive"):
        policy.update("a", [1, 2], .5, weight=0)

def test_partial_custom_generation_resumes_audited_questions(tmp_path):
    import generate_benchmark as generator
    sources = ["Acme revenue was 100 million.", "Beta revenue was 80 million."]
    evidence = [{"doc_id": "a", "quote": sources[0]}, {"doc_id": "b", "quote": sources[1]}]
    checkpoint = tmp_path / "partial.json"
    def response(model, system, user, **kwargs):
        if system.startswith("Audit"):
            return {"checks": [{"index": i, "valid": True, "reason": "supported"} for i, _ in enumerate(json.loads(user))]}
        category = next(c for c in generator.CATEGORIES if f"category '{c}'" in system)
        return {"questions": [{"question": category + "?", "category": category, "reference_answer": "Supported", "reasoning": "Evidence", "evidence": evidence}]} if category == "local" else {"questions": []}
    with patch.object(generator, "chat_json", side_effect=response):
        with pytest.raises(ValueError, match="incomplete"):
            generator.generate_benchmark_for_corpus(sources, ["a", "b"], 1, checkpoint_path=checkpoint)
    assert len(json.loads(checkpoint.read_text())["questions"]) == 1
    def finish(model, system, user, **kwargs):
        if system.startswith("Audit"):
            return {"checks": [{"index": i, "valid": True, "reason": "supported"} for i, _ in enumerate(json.loads(user))]}
        assert "category 'local'" not in system
        category = next(c for c in generator.CATEGORIES if f"category '{c}'" in system)
        return {"questions": [{"question": category + "?", "category": category, "reference_answer": "Supported", "reasoning": "Evidence", "evidence": evidence}]}
    with patch.object(generator, "chat_json", side_effect=finish):
        questions = generator.generate_benchmark_for_corpus(sources, ["a", "b"], 1, checkpoint_path=checkpoint)
    assert len(questions) == 4

def test_evidence_audit_context_comes_from_original_source():
    import generate_benchmark as generator
    text = "For year 2026, amounts in millions. Revenue was 100 million."
    q = {"question": "Revenue?", "category": "local", "reference_answer": "100 million", "reasoning": "Quote", "evidence": [{"doc_id": "a", "quote": "Revenue was 100 million.", "source_context": "Invented period 2099"}]}
    checked = generator.clean_question(q, "local", {"a": text}, set())
    assert checked["evidence"][0]["source_context"] == text

def test_api_pacing_spaces_request_starts():
    from app import llm_client
    with patch.object(config, "LLM_BACKEND", "api"), patch.object(config, "API_MIN_INTERVAL", 15), patch.object(llm_client, "_next_request_at", 0), patch.object(llm_client.time, "monotonic", return_value=0), patch.object(llm_client.time, "sleep") as sleep:
        llm_client.pace_request()
        llm_client.pace_request()
    sleep.assert_called_once_with(15)


def test_passage_ids_resolve_to_original_text_and_reject_unknown_ids():
    import generate_benchmark as generator
    texts = ["SOURCE: Acme/2020\n\nRevenue was $ 100 million.\n\nExpenses were $ 80 million.",
             "Beta revenue was $ 90 million."]
    context, catalog = generator.source_catalog(texts, ["a", "b"])
    assert len(context) <= config.QA_INPUT_CHARS
    for ref, evidence in catalog.items():
        assert ref in context
        assert evidence["quote"] in dict(zip(["a", "b"], texts))[evidence["doc_id"]]
    ref = next(iter(catalog))
    q = {"category": "local", "question": "Revenue?", "reference_answer": "100 million", "reasoning": "Source", "evidence_ids": [ref]}
    assert generator.resolve_evidence(q, catalog, [])["evidence"] == [catalog[ref]]
    errors = []
    assert generator.resolve_evidence({**q, "evidence_ids": ["invented"]}, catalog, errors) is None
    assert "passage IDs" in errors[0]


def test_id_generation_keeps_global_cross_document_requirement():
    import generate_benchmark as generator
    context, catalog = generator.source_catalog(["Acme revenue grew in 2020.\n\nAcme expenses grew in 2020.", "Beta revenue grew in 2020."], ["a", "b"])
    refs = [ref for ref, item in catalog.items() if item["doc_id"] == "a"]
    q = {"category": "global", "question": "Common growth theme?", "reference_answer": "Growth", "reasoning": "Compare", "evidence_ids": refs}
    errors = []
    resolved = generator.resolve_evidence(q, catalog, errors)
    assert generator.clean_question(resolved, "global", {"a": "Acme revenue grew in 2020.\n\nAcme expenses grew in 2020.", "b": "Beta revenue grew in 2020."}, set(), errors) is None
    assert "two documents" in errors[-1]


def test_v4_audited_checkpoint_resumes_after_passage_id_upgrade(tmp_path):
    import generate_benchmark as generator
    texts = ["Acme revenue was 100 million.", "Beta revenue was 80 million."]
    checkpoint = tmp_path / "partial.json"
    def reply(model, system, user, **kwargs):
        if system.startswith("Audit"):
            return {"checks": [{"index": i, "valid": True} for i, _ in enumerate(json.loads(user))]}
        category = next(c for c in generator.CATEGORIES if f"category '{c}'" in system)
        return {"questions": [{"question": category + "?", "category": category, "reference_answer": "Supported", "reasoning": "Source", "evidence": [{"doc_id": "a", "quote": texts[0]}, {"doc_id": "b", "quote": texts[1]}]}]}
    with patch.object(generator, "GENERATOR_VERSION", generator.COMPATIBLE_GENERATOR_VERSION), patch.object(generator, "chat_json", side_effect=reply):
        generator.generate_benchmark_for_corpus(texts, ["a", "b"], 1, checkpoint_path=checkpoint)
    with patch.object(generator, "chat_json", side_effect=AssertionError("saved audited questions should be reused")):
        resumed = generator.generate_benchmark_for_corpus(texts, ["a", "b"], 1, checkpoint_path=checkpoint)
    assert len(resumed) == 4


def test_retry_sources_rotate_away_from_used_documents_and_keep_ids_scoped():
    import generate_benchmark as generator
    texts = ["Acme revenue was 100 million.\n\nAcme costs were 70 million.",
             "Beta revenue was 90 million.\n\nBeta costs were 60 million.",
             "Gamma revenue was 80 million.\n\nGamma costs were 50 million."]
    _, catalog = generator.source_catalog(texts, ["a", "b", "c"])
    approved = [{"evidence": [{"doc_id": "a"}, {"doc_id": "a"}]}]
    context, active, assigned = generator.attempt_sources(catalog, ["a", "b", "c"], "multi_hop", 0, approved)
    assert assigned == ["b"]
    assert "Acme revenue" not in context and "Beta revenue" in context
    assert all(item["doc_id"] == "b" for item in active.values())
    _, _, next_assigned = generator.attempt_sources(catalog, ["a", "b", "c"], "multi_hop", 1, approved)
    assert next_assigned != assigned
    _, global_refs, global_assigned = generator.attempt_sources(catalog, ["a", "b", "c"], "global", 0, approved)
    assert len(global_assigned) == 2
    outside = next(ref for ref, item in catalog.items() if item["doc_id"] == "a")
    assert generator.resolve_evidence({"evidence_ids": [outside]}, active, []) is None


def test_repeated_question_retry_changes_sources_and_recovers():
    import generate_benchmark as generator
    texts = ["Acme revenue was 100 million.\n\nAcme costs were 70 million.",
             "Beta revenue was 90 million.\n\nBeta costs were 60 million."]
    calls = []
    counts = {}
    def reply(model, system, user, **kwargs):
        if system.startswith("Audit"):
            return {"checks": [{"index": i, "valid": True} for i, _ in enumerate(json.loads(user))]}
        category = next(c for c in generator.CATEGORIES if f"category '{c}'" in system)
        index = counts.get(category, 0)
        counts[category] = index + 1
        calls.append((category, user, kwargs["temperature"]))
        # The second attempt repeats the first task, then the third recovers.
        task = 0 if index < 2 else index
        return {"questions": [{"question": f"{category} task {task}?", "category": category,
                "reference_answer": "Supported", "reasoning": "Source", "evidence": [
                {"doc_id": "a", "quote": texts[0]}, {"doc_id": "b", "quote": texts[1]}]}]}
    with patch.object(generator, "chat_json", side_effect=reply):
        questions = generator.generate_benchmark_for_corpus(texts, ["a", "b"], 2)
    assert len(questions) == 8
    local = [(u, temp) for c, u, temp in calls if c == "local"]
    assert len(local) == 3
    assert local[0][0].splitlines()[0] != local[1][0].splitlines()[0]
    assert local[1][1] > local[0][1]
    assert "Repeated question" in local[2][0]


def test_incompatible_generation_checkpoint_is_not_overwritten(tmp_path):
    import generate_benchmark as generator
    path = tmp_path / "partial.json"
    path.write_text(json.dumps({"signature": "incompatible", "questions": [{"question": "Keep me"}]}))
    before = path.read_bytes()
    with patch.object(generator, "chat_json", side_effect=AssertionError("must not call model")):
        with pytest.raises(ValueError, match="Existing progress preserved"):
            generator.generate_benchmark_for_corpus(["Acme revenue was 100 million."], ["a"], 1, checkpoint_path=path)
    assert path.read_bytes() == before


def test_local_checkpoint_ignores_unused_hosted_endpoint_and_keeps_backup(tmp_path):
    import generate_benchmark as generator
    path = tmp_path / "partial.json"
    texts = ["Acme revenue was 100 million.", "Beta revenue was 90 million."]
    def reply(model, system, user, **kwargs):
        if system.startswith("Audit"):
            return {"checks": [{"index": i, "valid": True} for i, _ in enumerate(json.loads(user))]}
        category = next(c for c in generator.CATEGORIES if f"category '{c}'" in system)
        return {"questions": [{"question": category + "?", "category": category, "reference_answer": "Supported", "reasoning": "Source", "evidence": [{"doc_id": "a", "quote": texts[0]}, {"doc_id": "b", "quote": texts[1]}]}]}
    with patch.object(config, "API_BASE_URL", "https://integrate.api.nvidia.com/v1"), patch.object(generator, "chat_json", side_effect=reply):
        generator.generate_benchmark_for_corpus(texts, ["a", "b"], 1, checkpoint_path=path)
    with patch.object(config, "API_BASE_URL", "https://router.huggingface.co/v1"), patch.object(generator, "chat_json", side_effect=AssertionError("must reuse approved questions")):
        questions = generator.generate_benchmark_for_corpus(texts, ["a", "b"], 1, checkpoint_path=path)
    assert len(questions) == 4
    assert path.with_suffix(".previous.json").exists()


@pytest.mark.parametrize("version", ["balanced_evidence_v4", "balanced_evidence_v5"])
@pytest.mark.parametrize("old_endpoint", ["https://router.huggingface.co/v1", "https://integrate.api.nvidia.com/v1"])
def test_legacy_local_endpoint_fingerprint_migrates_without_model_calls(tmp_path, version, old_endpoint):
    import hashlib
    import generate_benchmark as generator
    texts = ["Acme revenue was 100 million.", "Beta revenue was 90 million."]
    ids = ["a", "b"]
    evidence = [{"doc_id": doc, "quote": text} for doc, text in zip(ids, texts)]
    questions = [{"question": category + "?", "category": category, "reference_answer": "Supported", "reasoning": "Source", "evidence": evidence} for category in generator.CATEGORIES]
    material = {"target": 1, "version": version, "question_model": config.QUESTION_MODEL,
                "judge_model": config.JUDGE_MODEL, "backend": "ollama", "endpoint": old_endpoint,
                "input_chars": config.QA_INPUT_CHARS}
    old_signature = generator.corpus_fingerprint(texts, ids) + ":" + hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()
    path = tmp_path / "partial.json"
    path.write_text(json.dumps({"signature": old_signature, "questions": questions}))
    with patch.object(config, "LLM_BACKEND", "ollama"), patch.object(generator, "chat_json", side_effect=AssertionError("reuse audited questions")):
        resumed = generator.generate_benchmark_for_corpus(texts, ids, 1, checkpoint_path=path)
    assert len(resumed) == 4
    assert path.with_suffix(".previous.json").exists()


def test_focused_source_budget_preserves_selected_document_context():
    import generate_benchmark as generator
    texts = ["First plan amendment reduced obligations.\n\n" + "Other background information. " * 70 + "\n\nSecond plan amendment affected income.",
             "Another company reported financial results."]
    with patch.object(config, "QA_INPUT_CHARS", 2400):
        _, catalog = generator.source_catalog(texts, ["a", "b"])
        context, active, assigned = generator.attempt_sources(catalog, ["a", "b"], "multi_hop", 0, [], sources=dict(zip(["a", "b"], texts)))
    assert assigned == ["a"]
    assert len(context) <= 2400
    assert "First plan amendment" in context and "Second plan amendment" in context
    assert all(e["quote"] in texts[0] for e in active.values())
