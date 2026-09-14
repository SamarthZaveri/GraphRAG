"""
Single source of truth for the engine-router's feature computation --
used both by RL/*.py during offline training (via a thin re-export shim
at RL/features.py) and by corpus_router.py at live query-serving time.

Previously this logic lived only in RL/features.py, which meant the live
app couldn't reuse it without backend importing from the RL/ folder --
backwards, since backend/app must run standalone and RL/ is meant to
depend on backend, not the other way around. This is the fix: the real
implementation lives here now.

REVISION HISTORY (kept from the original RL/features.py, since the same
reasoning still applies): started as a 6-feature vector including
community_modularity and rgcn_val_auc computed over the WHOLE graph. Both
saturated high (0.76-0.97) on almost every real corpus tested, including
"control" corpora with no real cross-document structure -- the table
parser creates dense intra-document structure regardless of whether
there's real structure connecting documents. Also tried and dropped:
doc_pair_connectivity (saturated at 1.0 for nearly all multi-doc corpora
-- real financial filings share generic boilerplate entities regardless
of relatedness) and cross_doc_subgraph_modularity (statistically unstable
on small subgraphs -- an unrelated-companies corpus scored on par with
genuine same-company-multi-quarter corpora).

What survived real-data validation, kept here:
  - cross_doc_entity_fraction: consistently separated genuinely-linked
    multi-quarter/same-company corpora from everything else.
  - entity_recurrence_depth: informative beyond raw doc count -- a 3-doc
    industry-cluster corpus caps below full recurrence (no single entity
    spans all 3 docs), a genuine multi-quarter corpus hits full recurrence.
  - num_docs_normalized, bias: kept as-is.

SECOND FINDING (12-corpus real evaluation): even these doc-level features,
correctly computed, don't predict engine performance well -- the two
corpora with the HIGHEST cross-document structure were both won by vector
RAG overall. Conclusion: which engine wins depends on the QUESTION asked,
not just the corpus. See build_query_context() below, which is what
corpus_router.route_query() actually uses for live routing decisions --
extract_features() below is doc-level only and is kept for descriptive
use (corpus_router.analyze_corpus(), the /api/corpus-analysis endpoint)
and as an input to build_query_context(), not as a routing decision by
itself anymore.
"""
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


def extract_features(store, modularity: float | None = None, rgcn_val_auc: float | None = None) -> np.ndarray:
    """
    Doc-level structural features only. `modularity` and `rgcn_val_auc`
    params are kept in the signature for call-site compatibility with
    RL/run_experiments.py but are IGNORED -- both were dropped from the
    vector after real-data validation (see module docstring).
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