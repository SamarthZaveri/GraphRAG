"""Analyze corpus structure and select a retrieval engine for each question.

A trained three-engine LinUCB policy uses document features and the predicted
question category. When no compatible policy exists, routing uses the corpus
recommendation. Benchmark comparisons always evaluate all three engines."""
from __future__ import annotations
import json
import re
import traceback
from typing import Optional, Tuple

from . import config, community, rgcn, router_features, query_engine, workspace
from .graph_store import GraphStore
from .query_bandit import LinUCBBandit

def _analysis_path():
    return workspace.state_dir() / "corpus_analysis.json"


# The router policy is a trained model shared by every corpus, so it stays in the shared state dir.
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
    with open(_analysis_path(), "w") as f:
        json.dump(analysis, f)


def load_analysis() -> Optional[dict]:
    if not _analysis_path().exists():
        return None
    with open(_analysis_path()) as f:
        return json.load(f)


def _fallback_to_corpus_analysis(prefix: str) -> dict:
    analysis = load_analysis() or analyze_corpus()
    return {"engine": analysis.get("recommended_engine", "graphrag"),
            "reason": f"[{prefix}] {analysis.get('reason', '')}", "source": "corpus_fallback"}


def canonical_model(name) -> str:
    """One identity for the same weights across providers: Ollama "qwen2.5:14b" and
    Hugging Face "Qwen/Qwen2.5-14B-Instruct" both become "qwen2.5-14b"."""
    name = str(name or "").lower().rsplit("/", 1)[-1].replace(":", "-")
    return re.sub(r"-(instruct|chat|it)$", "", name)


def _policy_matches_config(bandit: LinUCBBandit) -> bool:
    """The policy's rewards were earned by a specific retrieval pipeline and model, so serving must
    use the same ones. The transport (local Ollama vs a hosted API) may differ: same model family."""
    meta = bandit.metadata
    return (meta.get("retrieval_version") == config.RETRIEVAL_VERSION
            and canonical_model(meta.get("answer_model")) == canonical_model(config.ANSWER_MODEL)
            and canonical_model(meta.get("extraction_model")) == canonical_model(config.EXTRACTION_MODEL))


def policy_status() -> dict:
    """Whether a deployed router policy exists and matches the serving configuration."""
    if not QUERY_BANDIT_PATH.exists():
        return {"deployed": False, "compatible": False}
    try:
        bandit = LinUCBBandit.load(QUERY_BANDIT_PATH)
    except (OSError, ValueError, KeyError):
        return {"deployed": True, "compatible": False}
    exact = (bandit.metadata.get("answer_model") == config.ANSWER_MODEL
             and bandit.metadata.get("llm_backend") == config.LLM_BACKEND)
    return {"deployed": True, "compatible": _policy_matches_config(bandit), "exact_match": exact, "arms": bandit.arms,
            "metadata": {k: bandit.metadata.get(k) for k in ("dataset", "split", "answer_model", "retrieval_version")}}


def route_query(question: str) -> Tuple[str, str]:
    """Per-question engine decision. Returns (engine, human_readable_reason)."""
    decision = route_query_details(question)
    return decision["engine"], decision["reason"]


def route_query_details(question: str) -> dict:
    """
    Per-question engine decision with the router's evidence: engine, reason,
    source ("bandit", "corpus_fallback" or "empty"), and for the bandit the
    predicted question category and per-engine predicted rewards.

    Falls back to the static corpus-level recommendation (analyze_corpus)
    if: no documents are ingested, no trained query bandit exists yet
    (train and deploy the policy using scripts/run-local-flow.ps1 to produce one), or anything about
    classification/scoring fails -- a routing decision should degrade
    gracefully, not 500 the whole query.
    """
    store = GraphStore.load()
    if store is None or store.graph.number_of_nodes() == 0:
        return {"engine": "vector_rag", "reason": "No documents ingested yet.", "source": "empty"}

    if not QUERY_BANDIT_PATH.exists():
        return _fallback_to_corpus_analysis(
            "fallback: no trained query-level router yet, train and deploy the policy using scripts/run-local-flow.ps1"
        )

    try:
        doc_features = router_features.extract_features(store)
        category, _entities = query_engine.classify_query(question)
        context = router_features.build_query_context(doc_features, category)

        bandit = LinUCBBandit.load(QUERY_BANDIT_PATH)
        from .engines import ENGINE_NAMES
        if set(bandit.arms) != set(ENGINE_NAMES) or bandit.context_dim != router_features.QUERY_CONTEXT_DIM:
            return _fallback_to_corpus_analysis("fallback: retrain router for all three engines")
        if not _policy_matches_config(bandit):
            return _fallback_to_corpus_analysis("fallback: router model/retrieval configuration changed; retrain")
        # Serving time uses pure exploitation (predicted_reward), not
        # select_arm's UCB exploration bonus -- a live request isn't an
        # opportunity to explore, it needs the best current estimate.
        rewards = {arm: bandit.predicted_reward(arm, context) for arm in bandit.arms}
        engine = max(rewards, key=rewards.get)
        reason = (f"question classified as '{category}'; predicted reward "
                  f"graphrag={rewards.get('graphrag', 0):.2f}, "
                  f"vector_rag={rewards.get('vector_rag', 0):.2f}, "
                  f"hybrid_rag={rewards.get('hybrid_rag', 0):.2f}")
        return {"engine": engine, "reason": reason, "source": "bandit", "category": category,
                "predicted_rewards": {arm: round(r, 4) for arm, r in rewards.items()},
                "policy": {k: bandit.metadata.get(k) for k in ("dataset", "split", "dataset_fingerprint")}}
    except Exception:
        traceback.print_exc()
        return _fallback_to_corpus_analysis("fallback: query router error, see server logs")
