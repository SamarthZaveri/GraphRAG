"""Shared sources, numeric scoring, local profiles and mixed reward coverage."""
import json
from unittest.mock import Mock, patch
import pytest
from numeric_scoring import parse_scalar, score_numeric
from prepare_shared_benchmark import prepare
from prepare_finqa import prepare as prepare_finqa
from train_shared_router import load_weighted
from app import config, benchmark, engines


@pytest.mark.parametrize("answer,gold,human,program,correct", [
    ("FINAL_NUMERIC_ANSWER: 53 percent", ".53232", "53%", "divide(14001, 26302)", True),
    ("FINAL_NUMERIC_ANSWER: -3.2 percent", "-.03219", "-3.2%", "subtract(959.2,991.1), divide(#0,991.1)", True),
    ("FINAL_NUMERIC_ANSWER: 53.2 percent", "53.2", "53.2%", "divide(1,2), multiply(#0,const_100)", True),
    ("FINAL_NUMERIC_ANSWER: (1,234)", "-1234", "-1234", "", True),
    ("FINAL_NUMERIC_ANSWER: 8", "12", "12", "", False),
    ("Revenue was 10 in 2020 and 12 in 2021", "12", "12", "", False),
    ("FINAL_NUMERIC_ANSWER: no", "no", "no", "", True),
    ("FINAL_NUMERIC_ANSWER: nan", "12", "12", "", False),
    ("FINAL_NUMERIC_ANSWER: 1\nFINAL_NUMERIC_ANSWER: 12", "12", "12", "", False),
    # v2: bare percent for a ratio gold, descriptive units, the filing's own scale.
    ("FINAL_NUMERIC_ANSWER: 5.42", "0.05418", "5.4%", "divide(338, 6239)", True),
    ("FINAL_NUMERIC_ANSWER: 0.0542", "0.05418", "5.4%", "divide(338, 6239)", True),
    ("FINAL_NUMERIC_ANSWER: 15.04", "0.20997", "21%", "divide(1, 2)", False),
    ("FINAL_NUMERIC_ANSWER: 5.42", "0.05418", "0.05418", "", False),
    ("FINAL_NUMERIC_ANSWER: 71 thousand barrels per day", "71.0", "71", "", True),
    ("FINAL_NUMERIC_ANSWER: -132 mmboe", "-132.0", "-132", "", True),
    ("FINAL_NUMERIC_ANSWER: 72 thousand barrels per day", "71.0", "71", "", False),
    ("FINAL_NUMERIC_ANSWER: 71 in 2004", "71.0", "71", "", False),
])
def test_numeric_gold_scoring(answer, gold, human, program, correct):
    result = score_numeric(answer, {"reference_answer": gold, "human_answer": human, "reference_program": program, "question": "What is the result?"})
    assert (result["score"] == 5) == correct


def test_numeric_scale_conversion():
    q = {"reference_answer": "1200", "human_answer": "1200", "question": "What is revenue in millions?"}
    assert score_numeric("FINAL_NUMERIC_ANSWER: 1.2 billion", q)["score"] == 5
    assert score_numeric("FINAL_NUMERIC_ANSWER: 1.2 million", q)["score"] == 0


def example(report):
    return {"id": report + "/page.pdf-0", "pre_text": ["Revenue was 10 and profit was 2."], "post_text": [],
            "table": [["Metric", "2020"], ["Revenue", "10"]],
            "qa": {"question": "Revenue?", "exe_ans": 10, "answer": "10", "program": "", "gold_inds": {"text_0": "Revenue was 10 and profit was 2."}}}


def test_shared_questions_use_exact_same_sources_and_splits(tmp_path):
    source = tmp_path / "finqa"
    original = prepare_finqa({s: [example(report)] for s, report in zip(("train", "dev", "test"), ("A/2020", "B/2020", "C/2020"))}, source, {s: 1 for s in ("train", "dev", "test")})
    def generate(texts, ids, count, **kwargs):
        return [{"id": str(i), "category": c, "question": c + "?", "reference_answer": "Supported", "evidence": []} for i, c in enumerate(("local", "global", "multi_hop", "conflict"))]
    with patch("generate_benchmark.generate_benchmark_for_corpus", side_effect=generate):
        shared = prepare(source, tmp_path / "shared", per_category=3)
    for split in original["splits"]:
        old = original["splits"][split]["bundles"][0]
        new = shared["splits"][split]["bundles"][0]
        assert new["doc_ids"] == old["doc_ids"]
        assert new["questions"][0]["task"] == "numeric"
        assert all(q["task"] == "text" for q in new["questions"][1:])
        for doc in new["doc_ids"]:
            assert (source / "corpora" / old["id"] / (doc + ".txt")).read_bytes() == (tmp_path / "shared/corpora" / new["id"] / (doc + ".txt")).read_bytes()


def test_shared_numeric_bypasses_judge_and_exports_train_rewards(tmp_path, monkeypatch):
    import sys
    import numpy as np
    import run_public_benchmark as runner
    from app import main, query_engine, router_features
    from app.graph_store import GraphStore
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    docs = dataset / "corpora/train_0000"
    docs.mkdir(parents=True)
    (docs / "a.txt").write_text("Revenue was 10.")
    questions = [{"id": "numeric", "task": "numeric", "question": "Revenue?", "reference_answer": "10", "human_answer": "10"}]
    manifest = {"dataset": "shared_finqa", "prepared": True, "fingerprint": "f", "source_revision": "s", "splits": {"train": {"questions": 1, "bundles": [{"id": "train_0000", "doc_ids": ["a"], "questions": questions, "fingerprint": "b"}]}}}
    (dataset / "manifest.json").write_text(json.dumps(manifest))
    output = tmp_path / "run"
    runtime = output / "runtime"
    runtime.mkdir(parents=True)
    monkeypatch.setattr(config, "DATA_DIR", runtime)
    monkeypatch.setenv("LEDGER_DATA_DIR", str(runtime))
    monkeypatch.setattr(sys, "argv", ["runner", "--dataset-dir", str(dataset), "--split", "train", "--output-dir", str(output)])
    response = Mock(answer="FINAL_NUMERIC_ANSWER: 10", model_dump=Mock(return_value={"answer": "10"}))
    with patch.object(main, "_ingest_documents"), patch.object(GraphStore, "load"), patch.object(router_features, "extract_features", return_value=np.array([0, .1, .1, 1])), patch.object(query_engine, "classify_query", return_value=("local", [])), patch.object(engines, "answer_question", return_value=response), patch.object(benchmark, "judge_answers", side_effect=AssertionError("numeric must not call judge")):
        runner.main()
    summary = json.loads((output / "train/summary.json").read_text())
    assert summary["by_task"]["numeric"]["engine_rewards"]["graphrag"] == 1


def test_shared_weights_balance_numeric_and_four_text_categories(tmp_path):
    meta = {**benchmark.metadata(), "dataset": "shared_finqa", "split": "train", "dataset_fingerprint": "shared"}
    rows = [{"task": "numeric" if i < 3 else "text", "benchmark_category": c, "category": "local", "context": [0]*8,
             "metadata": meta, **{"reward_" + e: .5 for e in engines.ENGINE_NAMES}}
            for i, c in enumerate(("local", "local", "local", "local", "global", "multi_hop", "conflict"))]
    path = tmp_path / "query_results.jsonl"
    path.write_text("".join(json.dumps(r)+"\n" for r in rows))
    (tmp_path / "summary.json").write_text('{"complete":true}')
    weighted = load_weighted(path)
    total = sum(r["sample_weight"] for r in weighted)
    assert sum(r["sample_weight"] for r in weighted if r["task"] == "numeric") == pytest.approx(total*.5)


@pytest.mark.parametrize("problem,message", [
    ("held_out", "only train"),
    ("incomplete", "incomplete"),
    ("mixed_models", "Mixed model"),
    ("missing_engine", "reward"),
    ("invalid_context", "context"),
    ("missing_category", "all four"),
    ("obsolete_dataset", "shared-source"),
])
def test_shared_training_rejects_incompatible_rewards(tmp_path, problem, message):
    meta = {**benchmark.metadata(), "dataset": "shared_finqa", "split": "train", "dataset_fingerprint": "shared"}
    rows = [{"task": "numeric" if category == "numeric" else "text", "benchmark_category": category,
             "context": [0] * 8, "metadata": dict(meta),
             **{"reward_" + e: .5 for e in engines.ENGINE_NAMES}}
            for category in ("numeric", "local", "global", "multi_hop", "conflict")]
    if problem == "held_out":
        rows[0]["metadata"]["split"] = "test"
    elif problem == "mixed_models":
        rows[0]["metadata"]["answer_model"] = "different-model"
    elif problem == "missing_engine":
        rows[0].pop("reward_hybrid_rag")
    elif problem == "invalid_context":
        rows[0]["context"][0] = float("nan")
    elif problem == "missing_category":
        rows.pop()
    elif problem == "obsolete_dataset":
        for row in rows:
            row["metadata"]["dataset"] = "combined"
    path = tmp_path / "query_results.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "summary.json").write_text(json.dumps({"complete": problem != "incomplete"}))
    with pytest.raises(ValueError, match=message):
        load_weighted(path)


def test_rescore_resigns_numeric_checkpoints_and_keeps_text_scores(tmp_path):
    import hashlib
    from rescore_numeric import rescore_split
    from run_public_benchmark import checkpoint_signature, metric_name
    bundle = {"id": "b0", "fingerprint": "f0", "questions": [
        {"id": "num", "question": "What was the return on assets?", "reference_answer": "0.05418", "human_answer": "5.4%", "reference_program": "divide(338, 6239)", "task": "numeric"},
        {"id": "txt", "question": "Why?", "reference_answer": "Because.", "task": "text"}]}
    manifest = {"dataset": "shared_finqa", "splits": {"dev": {"bundles": [bundle]}}}
    checkpoint_dir = tmp_path / "dev" / "questions"
    checkpoint_dir.mkdir(parents=True)
    old_meta = {"metric": "shared_numeric_scalar_v1_text_judge_v1"}
    for q, answer, reward in ((bundle["questions"][0], "FINAL_NUMERIC_ANSWER: 5.42", 0.), (bundle["questions"][1], "Because.", .8)):
        record = {"task": q["task"], "metadata": old_meta, "answers": {e: {"answer": answer} for e in engines.ENGINE_NAMES},
                  "judge": {e: {"score": reward * 5} for e in engines.ENGINE_NAMES}, **{"reward_" + e: reward for e in engines.ENGINE_NAMES}}
        path = checkpoint_dir / (hashlib.sha256(q["id"].encode()).hexdigest() + ".json")
        path.write_text(json.dumps({"signature": checkpoint_signature(old_meta, bundle), "record": record}))
    stats = rescore_split(manifest, checkpoint_dir)
    assert stats["checkpoints"] == 2 and len(stats["flipped"]) == len(engines.ENGINE_NAMES)
    saved = {p.name: json.loads(p.read_text()) for p in checkpoint_dir.glob("*.json")}
    num, txt = (saved[hashlib.sha256(i.encode()).hexdigest() + ".json"] for i in ("num", "txt"))
    new_meta = {"metric": metric_name("shared_finqa")}
    assert num["signature"] == txt["signature"] == checkpoint_signature(new_meta, bundle)
    assert num["record"]["reward_hybrid_rag"] == 1. and txt["record"]["reward_hybrid_rag"] == .8
    assert rescore_split(manifest, checkpoint_dir)["already_current"] == 2
