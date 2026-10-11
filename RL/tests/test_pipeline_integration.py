"""Offline checks for graph feature extraction and malformed question output."""
import sys
from pathlib import Path

import numpy as np
import pytest



def test_feature_extraction_shape_and_bounds():
    from app.graph_store import GraphStore
    from app.router_features import extract_features, CONTEXT_DIM

    store = GraphStore()
    store.graph.add_node("A", type="Company", description="", source_docs=["doc1", "doc2"])
    store.graph.add_node("B", type="Metric", description="", source_docs=["doc1"])
    store.graph.add_node("C", type="Metric", description="", source_docs=["doc2"])

    features = extract_features(store)
    assert features.shape == (CONTEXT_DIM,)
    assert np.all(features >= 0.0) and np.all(features <= 1.0), "all features should be normalized to [0,1]"
    assert features[-1] == 1.0, "bias term should be exactly 1.0"

    assert features[1] == 0.2  # max document recurrence / 10
    assert features[2] == 0.2  # two documents / 10


def test_generate_benchmark_filters_malformed_output():
    from generate_benchmark import clean_question
    sources = {"doc1": "Revenue was 100 million in Q2 2025."}
    assert clean_question("not a dict", "local", sources, set()) is None
    assert clean_question({"question": "Q?", "category": "not_a_real_category", "reference_answer": "A"}, "local", sources, set()) is None
    assert clean_question({"question": "", "category": "local"}, "local", sources, set()) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
