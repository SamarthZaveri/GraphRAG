"""Offline checks for source isolation, split integrity and frozen evaluation."""
import json
import sys
from pathlib import Path
from unittest.mock import patch
import pytest
from prepare_finqa import prepare, render_document
from run_public_benchmark import choose_frozen
from app.query_bandit import LinUCBBandit


def example(report, index=0):
    return {"id": report + "/page_1.pdf-" + str(index), "pre_text": ["Revenue was 10."],
            "post_text": [], "table": [["metric", "2020", "2021"], ["Revenue", "10", "12"]],
            "qa": {"question": "Revenue growth?", "exe_ans": 20, "answer": "20%",
                   "program": "divide(subtract(12,10),10)", "gold_inds": {"text_0": "Revenue was 10."}},
            "model_input": "SECRET_GOLD_RETRIEVAL"}


def test_source_allowlist_and_table_headers():
    text = render_document(example("ACME/2020"))
    assert "2020: 10" in text and "2021: 12" in text
    assert "Revenue growth?" not in text and "divide(" not in text
    assert "SECRET_GOLD" not in text and "20%" not in text


def test_preparation_report_disjoint_splits(tmp_path):
    raw = {"train": [example("A/2020")], "dev": [example("A/2020", 1), example("B/2020")],
           "test": [example("B/2020", 2), example("C/2020")]}
    saved = prepare(raw, tmp_path, {s: 1 for s in raw})
    groups = [{q["report_id"] for b in saved["splits"][s]["bundles"] for q in b["questions"]} for s in raw]
    assert groups[0].isdisjoint(groups[1]) and groups[1].isdisjoint(groups[2])
    assert (tmp_path / "manifest.json").exists()


def test_frozen_selection_never_updates_policy():
    policy = LinUCBBandit(2, ["a", "b"])
    before = {arm: policy.A[arm].copy() for arm in policy.arms}
    with patch.object(policy, "update", side_effect=AssertionError("must remain frozen")):
        assert choose_frozen(policy, [0, 1]) in policy.arms
    for arm in policy.arms:
        assert (before[arm] == policy.A[arm]).all()


def test_training_rejects_dev_rewards(tmp_path):
    import train_shared_router as trainer
    from app import config
    row = {"context": [0] * 8, "metadata": {"dataset": "finqa", "split": "dev", "retrieval_version": config.RETRIEVAL_VERSION},
           **{"reward_" + a: .5 for a in trainer.ARMS}}
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="only train"):
        trainer.load_results(path)


def test_malformed_extraction_retries_then_caches(tmp_path):
    from app import extraction, config
    chunk = {"doc_id": "a", "chunk_id": "a::0", "text": "No financial facts."}
    with patch.object(config, "DATA_DIR", tmp_path), patch.object(extraction, "chat_json", side_effect=[{}, {"entities": [], "triples": []}]) as chat:
        result = extraction.extract_from_chunk(chunk)
    assert result.entities == [] and result.triples == []
    assert chat.call_count == 2
    assert "previous response" in chat.call_args.args[1]



def test_public_runner_train_and_frozen_dev_end_to_end(tmp_path, monkeypatch):
    import os
    import numpy as np
    import run_public_benchmark as runner
    from app import config, benchmark, engines, main, query_engine, router_features
    from app.graph_store import GraphStore
    from unittest.mock import Mock
    dataset = tmp_path / "dataset"
    prepared = prepare({s: [example(name)] for s, name in zip(("train", "dev", "test"), ("A/2020", "B/2020", "C/2020"))}, dataset, {s: 1 for s in ("train", "dev", "test")})
    output = tmp_path / "results"
    runtime = output / "runtime"
    runtime.mkdir(parents=True)
    monkeypatch.setattr(config, "DATA_DIR", runtime)
    monkeypatch.setenv("LEDGER_DATA_DIR", str(runtime))
    scores = {e: {"score": i+1, "rationale": "Checked reference"} for i, e in enumerate(engines.ENGINE_NAMES)}
    response = Mock(model_dump=Mock(return_value={"answer": "20%", "citations": []}))
    args = ["runner", "--dataset-dir", str(dataset), "--output-dir", str(output)]
    with patch.object(main, "_ingest_documents"), patch.object(GraphStore, "load"), patch.object(router_features, "extract_features", return_value=np.array([0, .1, .1, 1])), patch.object(query_engine, "classify_query", return_value=("multi_hop", ["A"])), patch.object(engines, "answer_question", return_value=response) as answer, patch.object(benchmark, "judge_answers", return_value=scores) as judge:
        monkeypatch.setattr(sys, "argv", args + ["--split", "train"])
        runner.main()
        # Every engine answers each question; only GraphRAG reuses the single classification.
        assert answer.call_count == len(engines.ENGINE_NAMES)
        graph_call = next(c for c in answer.call_args_list if c.args[0] == "graphrag")
        assert graph_call.kwargs["classification"] == ("multi_hop", ["A"])
        train = json.loads((output / "finqa" / "train" / "query_results.jsonl").read_text())
        policy = LinUCBBandit(8, list(engines.ENGINE_NAMES))
        policy.metadata = train["metadata"]
        policy_path = tmp_path / "policy.json"
        policy.save(policy_path)
        before = policy_path.read_bytes()
        monkeypatch.setattr(sys, "argv", args + ["--split", "dev", "--policy", str(policy_path)])
        runner.main()
        assert json.loads((output / "finqa" / "dev" / "summary.json").read_text())["complete"]
        assert policy_path.read_bytes() == before
        assert not (output / "finqa" / "dev" / "query_results.jsonl").exists()
        calls = (answer.call_count, judge.call_count)
        runner.main()
        assert (answer.call_count, judge.call_count) == calls  # Matching checkpoints resume without LLM calls.



def test_training_rejects_incomplete_public_reward_collection(tmp_path):
    import train_shared_router as trainer
    from app import config
    row = {"context": [0] * 8, "metadata": {"dataset": "finqa", "split": "train", "retrieval_version": config.RETRIEVAL_VERSION},
           **{"reward_" + a: .5 for a in trainer.ARMS}}
    path = tmp_path / "query_results.jsonl"
    path.write_text(json.dumps(row))
    (tmp_path / "summary.json").write_text('{"complete": false}')
    with pytest.raises(ValueError, match="incomplete"):
        trainer.load_results(path)
