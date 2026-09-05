"""
Community detection over the knowledge graph + hierarchical summarization.

This is the core GraphRAG technique: cluster related entities, then ask the
LLM to write a summary of each cluster so "global" (broad/thematic) questions
can be answered by reading a handful of community summaries instead of the
entire graph or every chunk.

Tries Leiden (via igraph/leidenalg) at a small sweep of resolutions and
keeps whichever partition scores highest on modularity -- a standard,
interpretable measure of how well-separated the communities actually are,
rather than trusting one arbitrary fixed resolution. Falls back to
networkx's greedy-modularity communities if igraph/leidenalg aren't
importable in this environment. The chosen partition's modularity score is
returned alongside the membership mapping and persisted, since it's also
used downstream as one of the corpus-health signals for the GraphRAG-vs-
vector-RAG router.
"""
from __future__ import annotations
import json
from typing import Dict, List, Tuple

import networkx as nx

from . import config
from .graph_store import GraphStore
from .models import CommunitySummary
from .ollama_client import chat_json

LEIDEN_RESOLUTIONS_TO_TRY = [0.5, 0.75, 1.0, 1.25, 1.5]


def _build_undirected(graph: nx.MultiDiGraph) -> nx.Graph:
    undirected = nx.Graph()
    undirected.add_nodes_from(graph.nodes())
    for u, v in graph.edges():
        if undirected.has_edge(u, v):
            undirected[u][v]["weight"] += 1
        else:
            undirected.add_edge(u, v, weight=1)
    return undirected


def _modularity_of(undirected: nx.Graph, membership: Dict[str, int]) -> float:
    groups: Dict[int, set] = {}
    for node, cid in membership.items():
        groups.setdefault(cid, set()).add(node)
    communities = list(groups.values())
    if len(communities) < 2:
        return 0.0
    try:
        return nx.algorithms.community.modularity(undirected, communities, weight="weight")
    except Exception:
        return 0.0


def detect_communities(graph: nx.MultiDiGraph) -> Tuple[Dict[str, int], float]:
    """Returns ({node_name: community_id}, modularity) for the best-scoring
    partition found across a small sweep of Leiden resolutions."""
    undirected = _build_undirected(graph)
    if undirected.number_of_nodes() == 0:
        return {}, 0.0

    try:
        import igraph as ig
        import leidenalg

        idx_of = {n: i for i, n in enumerate(undirected.nodes())}
        names = list(undirected.nodes())
        edges = [(idx_of[u], idx_of[v]) for u, v in undirected.edges()]
        weights = [undirected[u][v]["weight"] for u, v in undirected.edges()]
        g = ig.Graph(n=len(names), edges=edges)
        g.es["weight"] = weights

        best_membership, best_modularity = None, -1.0
        for resolution in LEIDEN_RESOLUTIONS_TO_TRY:
            partition = leidenalg.find_partition(
                g, leidenalg.RBConfigurationVertexPartition,
                weights="weight", resolution_parameter=resolution,
            )
            membership = {names[i]: partition.membership[i] for i in range(len(names))}
            modularity = _modularity_of(undirected, membership)
            if modularity > best_modularity:
                best_membership, best_modularity = membership, modularity
        return best_membership, best_modularity
    except Exception:
        # Fallback: greedy modularity communities (stdlib networkx, no extra deps)
        communities = nx.algorithms.community.greedy_modularity_communities(undirected, weight="weight")
        membership = {}
        for cid, members in enumerate(communities):
            for m in members:
                membership[m] = cid
        return membership, _modularity_of(undirected, membership)


SUMMARY_SYSTEM_PROMPT = """You write concise analytical summaries of a cluster of related \
entities from a financial-report knowledge graph, for a system that answers broad, thematic \
questions ("global search"). Given a list of entities (with types/descriptions) and the \
relationship facts connecting them, write:

- "title": a short (<=8 word) label for what this community is about
- "summary": 3-6 sentences covering who/what is involved, the nature of the relationships \
  (metric values, period-over-period changes, segment/company structure), and anything a \
  financial analyst reviewing multiple filings would want to know about this cluster.

Be factual and grounded only in the provided facts. Return ONLY JSON, no preamble: \
{"title": str, "summary": str}
"""


def summarize_community(members: List[str], graph: nx.MultiDiGraph) -> dict:
    lines = []
    for n in members:
        data = graph.nodes[n]
        lines.append(f"- {n} ({data.get('type', 'Other')}): {data.get('description', '')}")
    lines.append("\nRelationships:")
    seen = set()
    for u, v, data in graph.edges(data=True):
        if u in members and v in members:
            key = (u, data.get("predicate"), v)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- {u} --[{data.get('predicate')}]--> {v} (source: {data.get('doc_id')})")

    result = chat_json(config.EXTRACTION_MODEL, SUMMARY_SYSTEM_PROMPT, "\n".join(lines), max_tokens=500)
    if not result:
        return {"title": f"Community ({len(members)} entities)", "summary": ""}
    return result


def build_community_summaries(store: GraphStore) -> List[CommunitySummary]:
    membership, modularity = detect_communities(store.graph)
    groups: Dict[int, List[str]] = {}
    for node, cid in membership.items():
        groups.setdefault(cid, []).append(node)

    summaries = []
    for cid, members in groups.items():
        if len(members) < 2:
            continue  # skip singleton "communities", not useful for global search
        result = summarize_community(members, store.graph)
        summaries.append(CommunitySummary(
            community_id=cid, level=0, members=members,
            title=result.get("title", f"Community {cid}"),
            summary=result.get("summary", ""),
        ))

    if len(summaries) > 1:
        summaries.append(build_root_summary(summaries))

    _save_modularity(modularity, num_communities=len(groups))
    return summaries


ROOT_SUMMARY_SYSTEM_PROMPT = """You write a single high-level synthesis across ALL of the \
community summaries from a financial-report knowledge graph, for a system that answers broad \
questions spanning multiple companies/periods ("global search"). Given a list of per-cluster \
summaries (each cluster covers a subset of entities — often one company, or one company's \
metrics for one period), write:

- "title": a short (<=8 word) label, e.g. "Cross-company Q2 2025 overview"
- "summary": 5-10 sentences that actually name and compare across the different \
  companies/entities covered by the clusters below — this is the one summary a reader would use \
  to answer a question like "compare X across all companies" or "summarize Y across the corpus", \
  so don't just concatenate the per-cluster summaries; actually synthesize across them (e.g. name \
  which company grew fastest, which had the largest risk factors, etc., wherever the per-cluster \
  summaries give you enough to say so).

Be factual and grounded only in the provided cluster summaries — don't invent comparisons the \
clusters don't support. Return ONLY JSON: {"title": str, "summary": str}
"""


def build_root_summary(leaf_summaries: List[CommunitySummary]) -> CommunitySummary:
    """A level-1 rollup over all leaf-level community summaries, so broad
    cross-entity questions ("compare X across all companies") have a summary
    to draw on even when the leaf communities are fragmented per-company or
    per-metric and no single leaf covers the whole comparison."""
    lines = [f"[{s.title}] (covers: {', '.join(s.members[:6])})\n{s.summary}" for s in leaf_summaries]
    result = chat_json(config.ANSWER_MODEL, ROOT_SUMMARY_SYSTEM_PROMPT, "\n\n".join(lines), max_tokens=700)
    all_members = sorted({m for s in leaf_summaries for m in s.members})
    if not result:
        return CommunitySummary(community_id=-1, level=1, members=all_members,
                                 title="Corpus overview", summary="")
    return CommunitySummary(
        community_id=-1, level=1, members=all_members,
        title=result.get("title", "Corpus overview"),
        summary=result.get("summary", ""),
    )


SUMMARIES_PATH = config.GRAPH_STATE_DIR / "community_summaries.json"
MODULARITY_PATH = config.GRAPH_STATE_DIR / "community_modularity.json"


def _save_modularity(modularity: float, num_communities: int):
    with open(MODULARITY_PATH, "w") as f:
        json.dump({"modularity": modularity, "num_communities": num_communities}, f)


def load_modularity() -> dict:
    if not MODULARITY_PATH.exists():
        return {"modularity": None, "num_communities": None}
    with open(MODULARITY_PATH) as f:
        return json.load(f)


def save_summaries(summaries: List[CommunitySummary]):
    with open(SUMMARIES_PATH, "w") as f:
        json.dump([s.model_dump() for s in summaries], f)


def load_summaries() -> List[CommunitySummary]:
    if not SUMMARIES_PATH.exists():
        return []
    with open(SUMMARIES_PATH) as f:
        data = json.load(f)
    return [CommunitySummary(**d) for d in data]