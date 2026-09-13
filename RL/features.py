"""
Extracts the bandit's context feature vector from an ingested corpus.

REVISION 2 (post real-data validation across two runs): cut down from the
6-feature v1 to a 4-feature set. Real data confirmed two of the six were
not carrying signal:

  - `doc_pair_connectivity` (dropped): read 1.0 for nearly every multi-doc
    corpus regardless of relatedness (saas_2025, airlines_2025, even
    control_mixed_unrelated_1 all hit 1.0) -- it was just re-encoding
    "has >=2 docs," which num_docs_normalized already covers.
  - `cross_doc_subgraph_modularity` (dropped): greedy modularity on a
    small subgraph (a handful of cross-doc nodes) is statistically
    unstable -- banks_2025, which has no reason to have real cross-doc
    structure, scored 0.667, on par with the genuine longitudinal
    corpora. Real data raised the same doubt flagged when this feature
    was designed, so it's cut rather than kept as a coin flip.

What's kept, because it held up across multiple real runs:
  - `cross_doc_entity_fraction`: consistently separated five9_longitudinal
    / maxlinear_longitudinal from every other corpus.
  - `entity_recurrence_depth`: diverges from num_docs_normalized in an
    informative way -- a 3-doc industry-cluster corpus caps at 0.2 (no
    entity spans all 3 docs), a 4-doc longitudinal corpus hits 0.4 (full
    recurrence). Not redundant with num_docs, unlike the dropped features.
  - `num_docs_normalized`, `bias`: kept as-is.

Rationale for cutting rather than continuing to add: at ~12 real training
corpora, a bandit's context dimensionality should stay small enough that
every feature earns its place with real evidence. Two speculative,
unproven features were actively hurting trust in the policy more than a
smaller, fully-evidenced set would.

Feature vector (4-dim, was 6-dim in the previous revision):
  [cross_doc_entity_fraction, entity_recurrence_depth,
   num_docs_normalized, bias]

BREAKING CHANGE: CONTEXT_DIM went from 6 to 4. Old 6-dim rows in
experiment_results.jsonl are NOT directly compatible -- but do NOT
re-run all corpora to fix this. The 6-dim feature order was
[cross_doc_entity_fraction, doc_pair_connectivity,
cross_doc_subgraph_modularity, entity_recurrence_depth,
num_docs_normalized, bias], so the new 4-dim vector is exactly indices
[0, 3, 4, 5] of the old one -- a pure slice, no new computation needed.
Use migrate_features_v2.py (companion script) to do this slice on the
existing recorded data in place, saving a full re-run.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import networkx as nx

BACKEND_ROOT = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

CONTEXT_DIM = 4
FEATURE_NAMES = [
    "cross_doc_entity_fraction",
    "entity_recurrence_depth",
    "num_docs_normalized",
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


def extract_features(store, modularity: float | None = None, rgcn_val_auc: float | None = None) -> np.ndarray:
    """
    `modularity` and `rgcn_val_auc` params are kept in the signature for
    call-site compatibility (run_experiments.py currently passes them in)
    but are IGNORED -- both the modularity-based and rgcn-based features
    were dropped from the vector after real-data validation. Harmless to
    keep passing them; safe to remove from call sites later.
    """
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


# ---------------------------------------------------------------------------
# Per-QUESTION context, for the query-level engine router.
#
# The corpus-level bandit above answers "should this whole corpus route to
# GraphRAG or vector RAG" -- and real 12-corpus data showed that's the wrong
# question: five9_longitudinal and maxlinear_longitudinal had the highest
# cross-doc structure of any corpus and vector RAG still won both, most
# likely because most of their individual questions were simple local
# lookups despite the corpus itself being complex. The engine that wins
# depends on the QUESTION, not just the corpus it came from.
#
# This context vector combines the doc-level structural features (proven
# useful above) with the question's own category -- local/global/multi_hop/
# conflict, the SAME taxonomy generate_benchmark.py already labels questions
# with and query_engine.classify_query() already predicts at inference time.
# No new classifier or labeling scheme was introduced; this reuses both.
# ---------------------------------------------------------------------------

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


def build_query_context(doc_features: np.ndarray, category: str) -> np.ndarray:
    """
    Combines a corpus's doc-level context (from extract_features, a 4-dim
    vector ending in its own bias term) with a one-hot encoding of this
    question's category, into one 8-dim vector for the per-question bandit.

    The doc-level vector's own bias term is dropped here (index 3) since
    the combined vector carries a single shared bias term at the end
    instead -- keeping two separate "always 1.0" bias entries would double
    -count the intercept for no benefit.

    Unrecognized categories fall back to "global" rather than raising, to
    match the same fallback behavior used elsewhere (generate_benchmark.py,
    classify_query) when a category is missing or malformed.
    """
    if category not in QUERY_CATEGORIES:
        category = "global"
    doc_part = [float(doc_features[0]), float(doc_features[1]), float(doc_features[2])]
    one_hot = [1.0 if category == c else 0.0 for c in QUERY_CATEGORIES]
    return np.array(doc_part + one_hot + [1.0], dtype=float)