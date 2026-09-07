"""
Corpus-health router: decides whether GraphRAG or vector RAG is better
suited to THIS specific ingested corpus, using simple, explainable rules
over signals already computed elsewhere in the pipeline (cross-document
entity overlap, community modularity, R-GCN validation AUC).

This is a rule-based v1 -- an explicit, auditable decision function, not a
black-box classifier, which is the right choice at this data scale (a
handful of corpora, a few dozen benchmark questions -- nowhere near enough
to safely train a model to make this decision). It's the sensible base
layer to graduate to a learned policy later (e.g. a contextual bandit using
benchmark scores as reward), once there's (corpus, engine, outcome) data
across multiple DIFFERENTLY-SHAPED corpora to learn from -- thresholds
tuned against one corpus shouldn't be mistaken for a validated model.

Only affects the single-answer "Ask" endpoint (/api/query) when the person
selects "Auto". Benchmark, Communities, and Knowledge Graph are unaffected
-- the benchmark specifically must keep comparing both engines head-to-head
regardless of what the router would pick, or it stops measuring anything.
"""
from __future__ import annotations
import json
from typing import Optional

from . import config, community, rgcn
from .graph_store import GraphStore

ANALYSIS_PATH = config.GRAPH_STATE_DIR / "corpus_analysis.json"

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