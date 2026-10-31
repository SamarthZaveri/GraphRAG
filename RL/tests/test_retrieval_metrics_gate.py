"""Ranked-retrieval metrics and the CI quality gate on frozen releases."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def test_ranking_scores_match_hand_computed_values():
    from analyze_results import ranking_scores
    s = ranking_scores(["x", "a", "y", "b"], {"a", "b"}, ks=(1, 3, 5))
    assert s["precision@1"] == 0 and s["recall@1"] == 0 and s["ndcg@1"] == 0
    assert s["precision@3"] == pytest.approx(1 / 3) and s["recall@3"] == 0.5
    # DCG@5 = 1/log2(3) + 1/log2(5); ideal = 1 + 1/log2(3)
    assert s["ndcg@5"] == pytest.approx((1 / 1.5849625 + 1 / 2.3219281) / (1 + 1 / 1.5849625), rel=1e-6)
    assert s["precision@5"] == pytest.approx(2 / 5)  # fewer than k returned still divides by k
    perfect = ranking_scores(["a", "b"], {"a", "b"}, ks=(3,))
    assert perfect["ndcg@3"] == pytest.approx(1.0) and perfect["recall@3"] == 1.0


def test_retrieval_metrics_dedupe_documents_and_skip_questions_without_gold():
    from analyze_results import retrieval_metrics
    cite = lambda *docs: {"citations": [{"doc_id": d} for d in docs]}
    records = [{"question_id": "q1", "answers": {"hybrid_rag": cite("a", "a", "b")}},
               {"question_id": "q2", "answers": {"hybrid_rag": cite("c")}}]
    out = retrieval_metrics(records, "hybrid_rag", {"q1": {"b"}})
    assert out["n"] == 1 and out["doc_mrr"] == 0.5 and out["precision@1"] == 0 and out["recall@3"] == 1.0


def _release(root, router=0.80, hybrid=0.79, significant_worse=False, tamper=False):
    rel = root / "releases" / "v1"
    (rel / "test").mkdir(parents=True)
    analysis = {"policies": {"always_graphrag": {"mean": 0.6, "n": 10}, "always_vector_rag": {"mean": 0.7, "n": 10},
                             "always_hybrid_rag": {"mean": hybrid, "n": 10}, "router": {"mean": router, "n": 10}},
                "router": {"vs_engines": {"hybrid_rag": {"mean_diff": router - hybrid, "ci95": [-0.1, -0.01],
                                                         "significant": significant_worse}}},
                "retrieval": {"hybrid_rag": {"doc_hit_rate": 0.95}}}
    (rel / "test" / "analysis.json").write_text(json.dumps(analysis))
    (rel / "test" / "summary.json").write_text(json.dumps({"complete": True, "failed_questions": []}))
    files = {f: hashlib.sha256((rel / f).read_bytes()).hexdigest() for f in ("test/analysis.json", "test/summary.json")}
    (rel / "MANIFEST.json").write_text(json.dumps({"frozen_at": "2026-10-31T00:00:00", "complete": {"test": True}, "files": files}))
    if tamper:
        (rel / "test" / "analysis.json").write_text(json.dumps({**analysis, "edited": True}))
    return rel


def test_quality_gate_passes_a_sound_release(tmp_path):
    import quality_gate
    failures, notes = quality_gate.check(_release(tmp_path), 0.02, 0.5)
    assert failures == [] and any("router 0.800" in n for n in notes)


@pytest.mark.parametrize("kwargs, message", [
    ({"router": 0.70}, "below hybrid_rag"),
    ({"router": 0.78, "significant_worse": True}, "significantly worse"),
    ({"tamper": True}, "changed since"),
])
def test_quality_gate_fails_regressions_and_tampering(tmp_path, kwargs, message):
    import quality_gate
    failures, _ = quality_gate.check(_release(tmp_path, **kwargs), 0.02, 0.5)
    assert any(message in f for f in failures)
