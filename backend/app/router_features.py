"""Document and question features shared by offline training and serving.

Document features measure cross-document entity fraction, maximum entity
recurrence and document count. Query contexts add four predicted category
indicators and a bias, giving eight features. Gold task labels and authored
categories never enter the serving context."""
from __future__ import annotations
import networkx as nx
import numpy as np

CONTEXT_DIM = 4
FEATURE_NAMES = [
    "cross_doc_entity_fraction",
    "entity_recurrence_depth",
    "num_docs_normalized",
    "bias",
]

QUERY_CATEGORIES = ["local", "global", "multi_hop", "conflict"]
QUERY_CONTEXT_DIM = 8  # 3 doc features (bias dropped) + 4 one-hot category + 1 combined bias
QUERY_FEATURE_NAMES = [
    "cross_doc_entity_fraction",
    "entity_recurrence_depth",
    "num_docs_normalized",
    "cat_local",
    "cat_global",
    "cat_multi_hop",
    "cat_conflict",
    "bias",
]


def _cross_doc_node_set(graph: nx.MultiDiGraph) -> set:
    """Entities whose source_docs span 2+ distinct documents."""
    return {n for n, d in graph.nodes(data=True) if len(d.get("source_docs", [])) >= 2}


def _entity_recurrence_depth(graph: nx.MultiDiGraph) -> float:
    """
    Normalized max number of distinct source docs any single entity
    appears in. A corpus of unrelated single-doc filings should score
    near 0 here; a same-company/N-quarters corpus should score high.
    Normalized by 10 (matches num_docs_normalized's cap) so a 10+-doc
    fully-recurring entity maxes out at 1.0.
    """
    if graph.number_of_nodes() == 0:
        return 0.0
    max_depth = max((len(d.get("source_docs", [])) for _, d in graph.nodes(data=True)), default=0)
    return min(max_depth / 10.0, 1.0)


def extract_features(store) -> np.ndarray:
    """Return normalized document structure and a bias term."""
    graph = store.graph
    doc_ids = store.ingested_doc_ids()

    cross_doc_nodes = _cross_doc_node_set(graph)
    total_entities = max(graph.number_of_nodes(), 1)
    cross_doc_entity_fraction = min(len(cross_doc_nodes) / total_entities, 1.0)

    entity_recurrence_depth = _entity_recurrence_depth(graph)

    num_docs = len(doc_ids)
    num_docs_norm = min(num_docs / 10.0, 1.0)

    return np.array([
        cross_doc_entity_fraction,
        entity_recurrence_depth,
        num_docs_norm,
        1.0,
    ], dtype=float)


def build_query_context(doc_features: np.ndarray, category: str) -> np.ndarray:
    """
    Combines a corpus's doc-level context (from extract_features, a 4-dim
    vector ending in its own bias term) with a one-hot encoding of a
    question's category (local/global/multi_hop/conflict -- the same
    taxonomy generate_benchmark.py labels training questions with and
    query_engine.classify_query() predicts for real questions at serving
    time), into one 8-dim vector for the per-question bandit.

    The doc-level vector's own bias term (index 3) is dropped here since
    the combined vector carries a single shared bias term at the end
    instead -- two separate "always 1.0" entries would double-count the
    intercept for no benefit.

    Unrecognized categories fall back to "global", matching the same
    fallback behavior used elsewhere (generate_benchmark.py, classify_query)
    when a category is missing or malformed.
    """
    if category not in QUERY_CATEGORIES:
        category = "global"
    doc_part = [float(doc_features[0]), float(doc_features[1]), float(doc_features[2])]
    one_hot = [1.0 if category == c else 0.0 for c in QUERY_CATEGORIES]
    return np.array(doc_part + one_hot + [1.0], dtype=float)