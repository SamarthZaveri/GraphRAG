"""
The query engine that makes this "GraphRAG" rather than vector RAG: for a
given question it decides whether to do

  - LOCAL search: find the specific entity/entities the question is about,
    walk 1-2 hops of the graph around them, and ground the answer in that
    neighborhood's edges + source chunks. Good for "what are X's obligations
    under the agreement with Y?"

  - GLOBAL search: retrieve the most relevant community summaries and
    synthesize across them. Good for broad/thematic questions like
    "summarize the key risks across all filings" that no single chunk answers.
"""
from __future__ import annotations
import json
from difflib import SequenceMatcher
from typing import List, Tuple

from . import config
from .community import load_summaries
from .extraction import get_client, _strip_json_fences
from .graph_store import GraphStore
from .models import Citation, QueryResponse, CommunitySummary


ROUTER_SYSTEM_PROMPT = """Classify a question about a set of contracts/financial filings as \
either "local" or "global":

- "local": the question is about specific named entities/parties/agreements and their direct \
  relationships or obligations (e.g. "What is the Sponsor's Success Fee?", "Who are the \
  parties to the Sponsor Support Agreement?").
- "global": the question requires synthesizing across many entities/documents or asks for a \
  broad theme, comparison, or summary (e.g. "What are the common obligations across all the \
  agreements?", "Which clauses conflict between the documents?", "Summarize the deal \
  structure.").

Also extract up to 4 key entity names mentioned or implied in the question (for local search \
seeding), even if you classify it as global.

Return ONLY JSON: {"mode": "local"|"global", "entities": [str, ...]}
"""


def route_question(question: str) -> Tuple[str, List[str]]:
    client = get_client()
    resp = client.messages.create(
        model=config.EXTRACTION_MODEL, max_tokens=300,
        system=ROUTER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
    )
    raw = _strip_json_fences("".join(b.text for b in resp.content if b.type == "text"))
    try:
        data = json.loads(raw)
        return data.get("mode", "global"), data.get("entities", [])
    except json.JSONDecodeError:
        return "global", []


def _fuzzy_match_nodes(names: List[str], store: GraphStore, top_n: int = 4) -> List[str]:
    all_nodes = list(store.graph.nodes())
    matched = []
    for name in names:
        scored = sorted(
            all_nodes, key=lambda n: SequenceMatcher(None, name.lower(), n.lower()).ratio(),
            reverse=True,
        )
        for cand in scored[:top_n]:
            if SequenceMatcher(None, name.lower(), cand.lower()).ratio() > 0.4 and cand not in matched:
                matched.append(cand)
                break
    return matched


def _local_context(question: str, store: GraphStore, seed_entities: List[str], hops: int = 1):
    seeds = _fuzzy_match_nodes(seed_entities, store)
    if not seeds:
        # fall back: crude keyword match against node names
        q_lower = question.lower()
        seeds = [n for n in store.graph.nodes() if n.lower() in q_lower or q_lower.find(n.lower()[:6]) != -1][:5]

    visited = set(seeds)
    frontier = set(seeds)
    for _ in range(hops):
        next_frontier = set()
        for n in frontier:
            if store.graph.has_node(n):
                next_frontier |= set(store.graph.successors(n)) | set(store.graph.predecessors(n))
        next_frontier -= visited
        visited |= next_frontier
        frontier = next_frontier

    facts = []
    citations: List[Citation] = []
    for u, v, data in store.graph.edges(data=True):
        if u in visited and v in visited:
            facts.append(f"- {u} --[{data.get('predicate')}]--> {v}  (source: {data.get('doc_id')}, evidence: {data.get('evidence')})")
            chunk = store.chunks.get(data.get("chunk_id"), {})
            citations.append(Citation(
                doc_id=data.get("doc_id", ""), chunk_id=data.get("chunk_id", ""),
                snippet=(chunk.get("text", "")[:300]),
            ))
    node_lines = []
    for n in visited:
        if store.graph.has_node(n):
            d = store.graph.nodes[n]
            node_lines.append(f"- {n} ({d.get('type')}): {d.get('description', '')}")

    return seeds, list(visited), node_lines, facts, citations[:8]


ANSWER_SYSTEM_PROMPT = """You are Ledger, a GraphRAG assistant answering questions about a set \
of contracts and financial filings using facts retrieved from a knowledge graph. Answer only \
from the provided entities/facts/summaries — if the retrieved context doesn't contain the \
answer, say so plainly rather than guessing. Cite which document(s) support each claim inline \
using the doc_id shown in the context (e.g. "(doc1_sponsor_support_agreement)"). Be precise \
about numbers, dates, and party names. Keep the answer focused and no longer than necessary."""


def answer_local(question: str) -> QueryResponse:
    store = GraphStore.load()
    if store is None or store.graph.number_of_nodes() == 0:
        return QueryResponse(question=question, mode_used="local",
                              answer="No documents have been ingested yet.", citations=[])
    mode, seed_entities = route_question(question)
    seeds, visited, node_lines, facts, citations = _local_context(question, store, seed_entities)

    context = "ENTITIES:\n" + "\n".join(node_lines) + "\n\nFACTS:\n" + "\n".join(facts)
    client = get_client()
    resp = client.messages.create(
        model=config.ANSWER_MODEL, max_tokens=800,
        system=ANSWER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Context from knowledge graph:\n{context}\n\nQuestion: {question}"}],
    )
    answer = "".join(b.text for b in resp.content if b.type == "text")
    return QueryResponse(question=question, mode_used="local", answer=answer,
                          citations=citations, graph_path=visited[:20])


def answer_global(question: str) -> QueryResponse:
    summaries = load_summaries()
    if not summaries:
        return QueryResponse(question=question, mode_used="global",
                              answer="No community summaries available yet — run ingestion first.",
                              citations=[])
    context = "\n\n".join(
        f"[Community: {s.title}] (members: {', '.join(s.members[:8])})\n{s.summary}" for s in summaries
    )
    client = get_client()
    resp = client.messages.create(
        model=config.ANSWER_MODEL, max_tokens=900,
        system=ANSWER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Community summaries:\n{context}\n\nQuestion: {question}"}],
    )
    answer = "".join(b.text for b in resp.content if b.type == "text")
    citations = [Citation(doc_id=s.title, chunk_id=f"community_{s.community_id}", snippet=s.summary[:300])
                 for s in summaries]
    return QueryResponse(question=question, mode_used="global", answer=answer, citations=citations)


def answer_question(question: str, mode: str = "auto") -> QueryResponse:
    if mode == "auto":
        routed_mode, _ = route_question(question)
    else:
        routed_mode = mode
    if routed_mode == "local":
        return answer_local(question)
    return answer_global(question)