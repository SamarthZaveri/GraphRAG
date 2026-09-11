"""
Extracts the bandit's context feature vector from an ingested corpus.

REVISION NOTE (post 12-corpus real run): the original 5-feature vector
included `community_modularity` and `rgcn_val_auc`, both computed over the
WHOLE graph. Both saturated high (0.76-0.97) on almost every corpus,
including "control" corpora with no real cross-document structure. Root
cause: the deterministic table parser creates dense, learnable structure
WITHIN every single document (metric->period chains), so whole-graph
modularity and whole-graph link-prediction AUC mostly measure "did the
table parser run," not "is there real structure connecting documents."

Only `cross_doc_entity_fraction` showed the expected pattern in that run
(~0.02-0.04 baseline, jumping to 0.184 for five9_longitudinal).

This revision:
  - keeps cross_doc_entity_fraction (it worked)
  - adds doc_pair_connectivity: fraction of DOCUMENT PAIRS (not entities)
    that share at least one entity. Deliberately a different axis from
    cross_doc_entity_fraction and entity_recurrence_depth -- asks "how
    broadly is the corpus interconnected" rather than "how much/how deep
    is any one overlap." A raw cross-doc-edge-fraction was considered and
    rejected: it's largely redundant with cross_doc_entity_fraction, and
    with only ~12 training corpora a near-duplicate feature wastes bandit
    capacity rather than adding signal.
  - adds cross_doc_subgraph_modularity: modularity computed ONLY on the
    subgraph induced by cross-doc entities and edges between them, so
    intra-document table structure can't inflate it. Uses NetworkX's
    greedy modularity communities directly on this filtered subgraph --
    no dependency on the Leiden pipeline in community.py.
  - adds entity_recurrence_depth: how many distinct source docs the most-
    recurring entity appears in, normalized. This is what should make a
    same-company/N-quarters corpus (e.g. five9_longitudinal) stand out
    from a set of unrelated single-doc corpora.
  - DROPS rgcn_val_auc entirely. Fixing it properly means restricting the
    R-GCN's own training/eval to cross-doc edges, which is a change to
    rgcn.py's training loop, not something fixable from this file alone.
    Flagged as follow-up work rather than guessed at.

Feature vector (6-dim, was 5-dim):
  [cross_doc_entity_fraction, doc_pair_connectivity,
   cross_doc_subgraph_modularity, entity_recurrence_depth,
   num_docs_normalized, bias]

BREAKING CHANGE: CONTEXT_DIM went from 5 to 4 (rgcn dropped) + 2 new = 6.
Anything that hardcodes vector length 5 (bandit.py weight init/shapes,
any saved trained_bandit.json from a previous run) will need to be
retrained from scratch -- old saved bandit state is NOT compatible with
this feature vector and should be deleted, not loaded.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import networkx as nx

BACKEND_ROOT = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

CONTEXT_DIM = 6
FEATURE_NAMES = [
    "cross_doc_entity_fraction",
    "doc_pair_connectivity",
    "cross_doc_subgraph_modularity",
    "entity_recurrence_depth",
    "num_docs_normalized",
    "bias",
]


def _cross_doc_node_set(graph: nx.MultiDiGraph) -> set:
    """Entities whose source_docs span 2+ distinct documents."""
    return {n for n, d in graph.nodes(data=True) if len(d.get("source_docs", [])) >= 2}


def _doc_pair_connectivity(graph: nx.MultiDiGraph, doc_ids: set) -> float:
    """
    Fraction of DOCUMENT PAIRS (not entities) that share at least one
    entity. Deliberately different axis from cross_doc_entity_fraction
    and entity_recurrence_depth: this asks "how broadly is the corpus
    interconnected" rather than "how much/how deep is any one overlap."
    A corpus where doc1<->doc2 and doc3<->doc4 share entities but doc1
    and doc3 never touch scores lower here than a fully-interconnected
    corpus, even if both could look similar on the other two features.
    Chosen over a raw cross-doc-edge-fraction because that feature is
    largely redundant with cross_doc_entity_fraction (edges mostly just
    connect the nodes that feature already counts) -- with only ~12
    training corpora, a near-duplicate feature wastes bandit capacity
    rather than adding signal.
    """
    docs = sorted(doc_ids)
    n = len(docs)
    if n < 2:
        return 0.0
    idx = {d: i for i, d in enumerate(docs)}
    connected_pairs = set()
    for _, d in graph.nodes(data=True):
        docs_here = [doc for doc in d.get("source_docs", []) if doc in idx]
        for i in range(len(docs_here)):
            for j in range(i + 1, len(docs_here)):
                a, b = idx[docs_here[i]], idx[docs_here[j]]
                connected_pairs.add((min(a, b), max(a, b)))
    total_pairs = n * (n - 1) / 2
    return min(len(connected_pairs) / total_pairs, 1.0)


def _cross_doc_subgraph_modularity(graph: nx.MultiDiGraph, cross_doc_nodes: set) -> float:
    """
    Modularity computed ONLY on the subgraph induced by cross-doc entities.
    Returns 0.0 if there aren't enough cross-doc nodes/edges to form at
    least 2 communities (modularity is undefined/meaningless below that,
    and this correctly reads as "no real cross-doc structure" rather than
    erroring or defaulting to something misleadingly high).
    """
    if len(cross_doc_nodes) < 2:
        return 0.0
    sub = graph.subgraph(cross_doc_nodes)
    # Collapse to a simple undirected graph for community detection --
    # MultiDiGraph parallel/directed edges aren't what greedy modularity
    # community detection expects, and direction doesn't matter for "is
    # there structure here" at this coarse a signal.
    simple = nx.Graph()
    simple.add_nodes_from(sub.nodes())
    for u, v in sub.edges():
        simple.add_edge(u, v)
    if simple.number_of_edges() == 0:
        return 0.0
    try:
        communities = nx.algorithms.community.greedy_modularity_communities(simple)
        if len(communities) < 2:
            return 0.0
        mod = nx.algorithms.community.modularity(simple, communities)
        return max(0.0, min(mod, 1.0))
    except Exception:
        # Degenerate graphs (e.g. all one component, disconnected edge
        # cases) -- treat as no measurable structure rather than crash.
        return 0.0


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
    but are IGNORED -- modularity is recomputed here on the cross-doc
    subgraph instead, and rgcn_val_auc is dropped. Callers can stop
    passing these once call sites are updated; harmless if left in place.
    """
    graph = store.graph
    cross_doc_nodes = _cross_doc_node_set(graph)
    doc_ids = store.ingested_doc_ids()

    total_entities = max(graph.number_of_nodes(), 1)
    cross_doc_entity_fraction = min(len(cross_doc_nodes) / total_entities, 1.0)

    doc_pair_connectivity = _doc_pair_connectivity(graph, doc_ids)
    cross_doc_subgraph_modularity = _cross_doc_subgraph_modularity(graph, cross_doc_nodes)
    entity_recurrence_depth = _entity_recurrence_depth(graph)

    num_docs = len(doc_ids)
    num_docs_norm = min(num_docs / 10.0, 1.0)

    return np.array([
        cross_doc_entity_fraction,
        doc_pair_connectivity,
        cross_doc_subgraph_modularity,
        entity_recurrence_depth,
        num_docs_norm,
        1.0,
    ], dtype=float)