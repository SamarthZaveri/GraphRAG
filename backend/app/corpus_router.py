"""
Two routing functions live here now, serving different purposes:

  - analyze_corpus() / load_analysis(): UNCHANGED from before. A rule-based,
    explainable, corpus-WIDE description of structure (cross-doc entity
    overlap, community modularity, R-GCN validation AUC). Still powers the
    descriptive /api/corpus-analysis endpoint, and still serves as the
    FALLBACK for route_query() below when no trained query-level bandit is
    available yet.

  - route_query(question): NEW. The actual per-QUESTION engine decision
    used by /api/query when engine="auto". Real 12-corpus evaluation data
    showed the corpus-wide recommendation above doesn't predict engine
    performance well -- the two corpora with the HIGHEST cross-document
    structure (same-company, multiple quarters) were both won by vector
    RAG overall, because most individual questions in them were still
    simple lookups. Which engine wins depends on the QUESTION, not just
    the corpus it came from. route_query() combines the corpus's doc-level
    structure with a live classification of the question itself
    (query_engine.classify_query: local/global/multi_hop/conflict) and
    scores both engines using a trained LinUCB bandit (see
    RL/train_query_bandit.py for training, backend/app/query_bandit.py for
    the implementation both training and serving share).

Only affects the single-answer "Ask" endpoint (/api/query) when the person
selects "Auto". Benchmark, Communities, and Knowledge Graph are unaffected
-- the benchmark specifically must keep comparing both engines head-to-head
regardless of what either router would pick, or it stops measuring anything.
"""
from __future__ import annotations
import json
import traceback
from typing import Optional, Tuple

from . import config, community, rgcn, router_features, query_engine
from .graph_store import GraphStore
from .query_bandit import LinUCBBandit

ANALYSIS_PATH = config.GRAPH_STATE_DIR / "corpus_analysis.json"
QUERY_BANDIT_PATH = config.GRAPH_STATE_DIR / "query_bandit.json"

MIN_SHARED_ENTITIES = 1
MIN_MODULARITY = 0.10
MIN_RGCN_AUC = 0.55


def analyze_corpus() -> dict:
    store = GraphStore.load()
    if store is None or store.graph.number_of_nodes() == 0:
        analysis = {
            "recommended_engine": "vector_rag",
            "reason": "No documents ingested yet.",
            "num_docs": 0, "shared_entity_count": 0,
            "community_modularity": None, "rgcn_val_auc": None,
        }
        _save(analysis)
        return analysis

    num_docs = len(store.ingested_doc_ids())
    shared_entity_count = sum(
        1 for _, data in store.graph.nodes(data=True) if len(data.get("source_docs", [])) >= 2
    )
    modularity_info = community.load_modularity()
    modularity = modularity_info.get("modularity")
    rgcn_metrics = rgcn.load_metrics()
    val_auc = rgcn_metrics.get("val_auc") if rgcn_metrics else None

    reasons = []
    recommend_graphrag = True

    if num_docs < 2:
        recommend_graphrag = False
        reasons.append(f"only {num_docs} document(s) ingested — no cross-document structure is possible")
    elif shared_entity_count < MIN_SHARED_ENTITIES:
        recommend_graphrag = False
        reasons.append("no entities are shared across documents, so the graph has no cross-document links to exploit")
    else:
        reasons.append(f"{shared_entity_count} entities appear in 2+ documents, giving the graph real cross-document structure")

    if modularity is not None:
        if modularity < MIN_MODULARITY:
            recommend_graphrag = False
            reasons.append(f"community modularity is low ({modularity:.2f}) — little coherent cluster structure for global search to summarize")
        else:
            reasons.append(f"community modularity is {modularity:.2f}, indicating real cluster structure")

    if val_auc is not None:
        if val_auc < MIN_RGCN_AUC:
            recommend_graphrag = False
            reasons.append(f"R-GCN validation AUC is low ({val_auc:.2f}, near chance) — graph structure isn't reliably learnable")
        else:
            reasons.append(f"R-GCN validation AUC is {val_auc:.2f}, indicating the graph structure is learnable/reliable")

    engine = "graphrag" if recommend_graphrag else "vector_rag"
    analysis = {
        "recommended_engine": engine,
        "reason": "; ".join(reasons),
        "num_docs": num_docs,
        "shared_entity_count": shared_entity_count,
        "community_modularity": modularity,
        "rgcn_val_auc": val_auc,
    }
    _save(analysis)
    return analysis


def _save(analysis: dict):
    config.GRAPH_STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(ANALYSIS_PATH, "w") as f:
        json.dump(analysis, f)


def load_analysis() -> Optional[dict]:
    if not ANALYSIS_PATH.exists():
        return None
    with open(ANALYSIS_PATH) as f:
        return json.load(f)


def _fallback_to_corpus_analysis(prefix: str) -> Tuple[str, str]:
    analysis = load_analysis() or analyze_corpus()
    engine = analysis.get("recommended_engine", "graphrag")
    reason = f"[{prefix}] {analysis.get('reason', '')}"
    return engine, reason


def route_query(question: str) -> Tuple[str, str]:
    """
    Per-question engine decision. Returns (engine, human_readable_reason).

    Falls back to the static corpus-level recommendation (analyze_corpus)
    if: no documents are ingested, no trained query bandit exists yet
    (run RL/train_query_bandit.py to produce one), or anything about
    classification/scoring fails -- a routing decision should degrade
    gracefully, not 500 the whole query.
    """
    store = GraphStore.load()
    if store is None or store.graph.number_of_nodes() == 0:
        return "vector_rag", "No documents ingested yet."

    if not QUERY_BANDIT_PATH.exists():
        return _fallback_to_corpus_analysis(
            "fallback: no trained query-level router yet, run RL/train_query_bandit.py"
        )

    try:
        doc_features = router_features.extract_features(store)
        category, _entities = query_engine.classify_query(question)
        context = router_features.build_query_context(doc_features, category)

        bandit = LinUCBBandit.load(QUERY_BANDIT_PATH)
        # Serving time uses pure exploitation (predicted_reward), not
        # select_arm's UCB exploration bonus -- a live request isn't an
        # opportunity to explore, it needs the best current estimate.
        rewards = {arm: bandit.predicted_reward(arm, context) for arm in bandit.arms}
        engine = max(rewards, key=rewards.get)
        reason = (f"question classified as '{category}'; predicted reward "
                  f"graphrag={rewards.get('graphrag', 0):.2f}, "
                  f"vector_rag={rewards.get('vector_rag', 0):.2f}")
        return engine, reason
    except Exception:
        traceback.print_exc()
        return _fallback_to_corpus_analysis("fallback: query router error, see server logs")