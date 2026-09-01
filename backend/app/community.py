"""
Community detection over the knowledge graph + hierarchical summarization.

This is the core GraphRAG technique: cluster related entities, then ask the
LLM to write a summary of each cluster so "global" (broad/thematic) questions
can be answered by reading a handful of community summaries instead of the
entire graph or every chunk.

Tries Leiden (via igraph/leidenalg) first, since that's what the original
GraphRAG paper uses; falls back to networkx's greedy-modularity communities
if igraph/leidenalg aren't importable in this environment.
"""
from __future__ import annotations
import json
from typing import Dict, List

import networkx as nx

from . import config
from .graph_store import GraphStore
from .models import CommunitySummary
from .ollama_client import chat_json


def detect_communities(graph: nx.MultiDiGraph) -> Dict[str, int]:
    """Returns {node_name: community_id} for the undirected projection of the graph."""
    undirected = nx.Graph()
    undirected.add_nodes_from(graph.nodes())
    for u, v in graph.edges():
        if undirected.has_edge(u, v):
            undirected[u][v]["weight"] += 1
        else:
            undirected.add_edge(u, v, weight=1)

    if undirected.number_of_nodes() == 0:
        return {}

    try:
        import igraph as ig
        import leidenalg

        idx_of = {n: i for i, n in enumerate(undirected.nodes())}
        names = list(undirected.nodes())
        edges = [(idx_of[u], idx_of[v]) for u, v in undirected.edges()]
        weights = [undirected[u][v]["weight"] for u, v in undirected.edges()]
        g = ig.Graph(n=len(names), edges=edges)
        g.es["weight"] = weights
        partition = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition,
            weights="weight", resolution_parameter=config.LEIDEN_RESOLUTION,
        )
        return {names[i]: partition.membership[i] for i in range(len(names))}
    except Exception:
        # Fallback: greedy modularity communities (stdlib networkx, no extra deps)
        communities = nx.algorithms.community.greedy_modularity_communities(undirected, weight="weight")
        result = {}
        for cid, members in enumerate(communities):
            for m in members:
                result[m] = cid
        return result


SUMMARY_SYSTEM_PROMPT = """You write concise analytical summaries of a cluster of related \
entities from a legal/financial knowledge graph, for a system that answers broad, thematic \
questions ("global search"). Given a list of entities (with types/descriptions) and the \
relationship facts connecting them, write:

- "title": a short (<=8 word) label for what this community is about
- "summary": 3-6 sentences covering who/what is involved, the nature of the relationships \
  (obligations, ownership, agreements, monetary terms), and anything a compliance/financial \
  analyst reviewing multiple contracts would want to know about this cluster.

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
    membership = detect_communities(store.graph)
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
    return summaries


SUMMARIES_PATH = config.GRAPH_STATE_DIR / "community_summaries.json"


def save_summaries(summaries: List[CommunitySummary]):
    with open(SUMMARIES_PATH, "w") as f:
        json.dump([s.model_dump() for s in summaries], f)


def load_summaries() -> List[CommunitySummary]:
    if not SUMMARIES_PATH.exists():
        return []
    with open(SUMMARIES_PATH) as f:
        data = json.load(f)
    return [CommunitySummary(**d) for d in data]
