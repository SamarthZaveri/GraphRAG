"""
Integration tests for the wiring around the bandit (not the bandit algorithm
itself, which test_bandit_synthetic.py already covers): feature extraction
from a real GraphStore, benchmark-generation output parsing, and the full
train_bandit.py training loop on realistic (but synthetic, so we know
ground truth) experiment data.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))


def test_feature_extraction_shape_and_bounds():
    from app.graph_store import GraphStore
    from features import extract_features, CONTEXT_DIM

    store = GraphStore()
    store.graph.add_node("A", type="Company", description="", source_docs=["doc1", "doc2"])
    store.graph.add_node("B", type="Metric", description="", source_docs=["doc1"])
    store.graph.add_node("C", type="Metric", description="", source_docs=["doc2"])

    features = extract_features(store, modularity=0.6, rgcn_val_auc=0.85)
    assert features.shape == (CONTEXT_DIM,)
    assert np.all(features >= 0.0) and np.all(features <= 1.0), "all features should be normalized to [0,1]"
    assert features[-1] == 1.0, "bias term should be exactly 1.0"

    # missing modularity/auc should fall back to sane defaults, not crash
    features2 = extract_features(store, modularity=None, rgcn_val_auc=None)
    assert np.allclose(features2, features)  # retired metrics do not affect features
    assert features2[1] == 0.2  # max document recurrence / 10
    assert features2[2] == 0.2  # two documents / 10


def test_generate_benchmark_filters_malformed_output():
    from generate_benchmark import clean_question
    sources = {"doc1": "Revenue was 100 million in Q2 2025."}
    assert clean_question("not a dict", "local", sources, set()) is None
    assert clean_question({"question": "Q?", "category": "not_a_real_category", "reference_answer": "A"}, "local", sources, set()) is None
    assert clean_question({"question": "", "category": "local"}, "local", sources, set()) is None


def test_full_training_loop_on_realistic_synthetic_data(tmp_path):
    """Builds experiment-result rows shaped like what run_experiments.py
    would actually produce -- favorable corpora (high cross-doc overlap,
    high entity recurrence and document count) should reward GraphRAG more; control
    corpora (single doc / no shared entities) should reward vector RAG
    more -- and checks the FULL train_bandit.py pipeline (not just bandit.py
    in isolation) learns to tell them apart."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import train_bandit
    importlib_reload = __import__("importlib").reload
    importlib_reload(train_bandit)

    rng = np.random.default_rng(42)
    rows = []
    for i in range(40):
        # favorable: high overlap/recurrence -> graphrag genuinely better
        rows.append({
            "corpus": f"favorable_{i}", "num_docs": 4,
            "features": [rng.uniform(0.4, 0.8), rng.uniform(0.3, 0.4), 0.4, 1.0],
            "reward_graphrag": rng.uniform(0.7, 0.85),
            "reward_vector_rag": rng.uniform(0.5, 0.65),
            "num_questions": 6,
        })
    for i in range(40):
        # control: low overlap/recurrence -> vector_rag genuinely better
        rows.append({
            "corpus": f"control_{i}", "num_docs": 1,
            "features": [rng.uniform(0.0, 0.1), 0.1, 0.1, 1.0],
            "reward_graphrag": rng.uniform(0.3, 0.5),
            "reward_vector_rag": rng.uniform(0.65, 0.85),
            "num_questions": 6,
        })

    results_path = tmp_path / "experiment_results.jsonl"
    with open(results_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    with patch.object(train_bandit, "RESULTS_PATH", results_path):
        loaded_rows = train_bandit.load_results()
        bandit, log = train_bandit.run_training(loaded_rows, alpha=1.0, seed=0)

    accuracy = np.mean([e["correct"] for e in log])
    assert accuracy >= 0.75, (
        f"trained policy should correctly distinguish favorable vs. control corpora most of the "
        f"time on data with this clear a signal (got {accuracy:.1%})"
    )

    # explicit recommendation check on the FINAL trained policy
    favorable_x = np.array([0.6, 0.4, 0.4, 1.0])
    control_x = np.array([0.05, 0.1, 0.1, 1.0])
    fav_arm, _ = bandit.select_arm(favorable_x)
    ctrl_arm, _ = bandit.select_arm(control_x)
    assert fav_arm == "graphrag"
    assert ctrl_arm == "vector_rag"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
