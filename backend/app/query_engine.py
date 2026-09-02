"""
The query engine that makes this "GraphRAG" rather than vector RAG.

For a given question it decides whether to do:

  - LOCAL search: find the specific entity/entities the question is about,
    walk 2 hops of the graph around them, PLUS pull in R-GCN-nearest nodes
    (entities close in embedding space even if not directly graph-adjacent —
    this catches relationships the extraction step failed to link
    explicitly), and ground the answer in that neighborhood's edges +
    the actual source chunk text (not just short evidence paraphrases).

  - GLOBAL search: retrieve the community summaries most relevant to the
    question (ranked by local text-embedding similarity, not dumped in
    wholesale) and synthesize across them.

Hybrid grounding: local search also pulls the same top-k raw chunks the
vector-RAG baseline would retrieve for this question and merges them into
the context. This means GraphRAG's context is always at least as grounded
in original text as the vector baseline's, on top of the graph structure —
if GraphRAG were losing to vector RAG mainly because compressed graph
triples threw away detail the raw prose had, this closes that gap directly.
"""
from __future__ import annotations
from difflib import SequenceMatcher
from typing import List, Tuple

from . import config
from .community import load_summaries
from .ollama_client import chat, chat_json
from .graph_store import GraphStore
from .models import Citation, QueryResponse
from . import rgcn

LOCAL_HOPS = 2
RGCN_EXPANSION_K = 4
GLOBAL_TOP_N_COMMUNITIES = 4
HYBRID_CHUNK_TOP_K = 4
HYBRID_VECTOR_FALLBACK_K = 4


ROUTER_SYSTEM_PROMPT = """Classify a question about a set of financial reports/filings as \
either "local" or "global":

- "local": the question is about specific named companies/metrics/periods and their direct \
  relationships or values (e.g. "What was MaxLinear's Q2 2025 net revenue?", "Who was appointed \
  Five9's CFO?").
- "global": the question requires synthesizing across many entities/documents or asks for a \
  broad theme, comparison, or summary (e.g. "Compare revenue growth across all three companies", \
  "Which companies improved margins this quarter?", "Summarize the key risks across all \
  filings.").

Also extract up to 4 key entity names mentioned or implied in the question (company names, \
metric names, periods) for local search seeding, even if you classify it as global.

Return ONLY JSON, no preamble: {"mode": "local"|"global", "entities": [str, ...]}
"""


def route_question(question: str) -> Tuple[str, List[str]]:
    data = chat_json(config.EXTRACTION_MODEL, ROUTER_SYSTEM_PROMPT, question, max_tokens=250)
    mode = data.get("mode", "global")
    if mode not in ("local", "global"):
        mode = "global"
    return mode, data.get("entities", []) or []


def _fuzzy_match_nodes(names: List[str], store: GraphStore, top_n: int = 4) -> List[str]:
    all_nodes = list(store.graph.nodes())
    matched = []
    for name in names:
        scored = sorted(
            all_nodes, key=lambda n: SequenceMatcher(None, name.lower(), n.lower()).ratio(),
            reverse=True,
        )
        best = scored[0] if scored else None
        best_score = SequenceMatcher(None, name.lower(), best.lower()).ratio() if best else 0.0
        if best is not None and best_score > 0.55 and best not in matched:
            matched.append(best)
            continue
        # weak/no string match — fall back to semantic linking via local text embeddings
        try:
            from .text_embeddings import top_k as embed_top_k
            node_texts = [f"{n} ({store.graph.nodes[n].get('type','')}): "
                          f"{store.graph.nodes[n].get('description','')}" for n in all_nodes]
            hits = embed_top_k(name, node_texts, k=1)
            if hits and hits[0][1] > 0.35:
                candidate = all_nodes[hits[0][0]]
                if candidate not in matched:
                    matched.append(candidate)
        except Exception:
            if best is not None and best not in matched:
                matched.append(best)
    return matched[:top_n]


def _graph_hop_expand(store: GraphStore, seeds: List[str], hops: int) -> set:
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
    return visited


def _local_context(question: str, store: GraphStore, seed_entities: List[str]):
    seeds = _fuzzy_match_nodes(seed_entities, store)
    if not seeds:
        q_lower = question.lower()
        seeds = [n for n in store.graph.nodes()
                 if n.lower() in q_lower or q_lower.find(n.lower()[:6]) != -1][:5]

    visited = _graph_hop_expand(store, seeds, LOCAL_HOPS)

    # R-GCN expansion: pull in nodes close in embedding space even if not
    # directly graph-adjacent, to bridge gaps left by imperfect extraction.
    rgcn_neighbors = rgcn.nearest_nodes(seeds, k=RGCN_EXPANSION_K, exclude=visited) if seeds else []
    visited |= set(rgcn_neighbors)

    facts = []
    citations: List[Citation] = []
    chunk_ids_seen = set()
    for u, v, data in store.graph.edges(data=True):
        if u in visited and v in visited:
            facts.append(f"- {u} --[{data.get('predicate')}]--> {v}  "
                          f"(source: {data.get('doc_id')}, evidence: {data.get('evidence')})")
            cid = data.get("chunk_id")
            if cid and cid not in chunk_ids_seen:
                chunk_ids_seen.add(cid)
                chunk = store.chunks.get(cid, {})
                citations.append(Citation(
                    doc_id=data.get("doc_id", ""), chunk_id=cid,
                    snippet=(chunk.get("text", "")[:300]),
                ))

    node_lines = []
    for n in visited:
        if store.graph.has_node(n):
            d = store.graph.nodes[n]
            tag = " [via R-GCN]" if n in rgcn_neighbors else ""
            node_lines.append(f"- {n} ({d.get('type')}){tag}: {d.get('description', '')}")

    # Hybrid grounding, always-on: pull the raw source-chunk text for the
    # facts found via graph traversal (capped)...
    raw_chunks = []
    for cid in list(chunk_ids_seen)[:HYBRID_CHUNK_TOP_K]:
        chunk = store.chunks.get(cid, {})
        if chunk.get("text"):
            raw_chunks.append(f"[{chunk.get('doc_id')} | {cid}]\n{chunk['text']}")

    # ...PLUS independently pull the same top-k chunks a vector-similarity
    # search over the question would return, regardless of whether graph
    # traversal found anything. This is the actual hybrid-retrieval safety
    # net: if entity linking fails or the graph has a gap, GraphRAG still
    # gets grounded in the same raw text the vector baseline would use,
    # instead of answering from a possibly-empty facts list.
    try:
        from . import vector_baseline
        collection = vector_baseline.get_collection()
        if collection.count() > 0:
            results = collection.query(query_texts=[question], n_results=min(HYBRID_VECTOR_FALLBACK_K, collection.count()))
            for cid, doc_text, meta in zip(results["ids"][0], results["documents"][0], results["metadatas"][0]):
                if cid in chunk_ids_seen:
                    continue  # already included above
                raw_chunks.append(f"[{meta.get('doc_id')} | {cid}]\n{doc_text}")
                chunk_ids_seen.add(cid)
                citations.append(Citation(doc_id=meta.get("doc_id", ""), chunk_id=cid, snippet=doc_text[:300]))
    except Exception:
        pass  # vector baseline not available/ingested yet — graph-only context still works

    return seeds, list(visited), node_lines, facts, citations[:14], raw_chunks


ANSWER_SYSTEM_PROMPT = """You are Ledger, a GraphRAG assistant answering questions about a set \
of financial reports using facts retrieved from a knowledge graph, plus the original source text \
those facts were extracted from. Answer only from the provided context — if it doesn't contain \
the answer, say so plainly rather than guessing. Cite which document(s) support each claim \
inline using the doc_id shown in the context (e.g. "(doc1_maxlinear_q2_2025_earnings)"). Be \
precise about numbers, dates, and company names — copy figures exactly as given.

CRITICAL: if the source text already states a percentage, growth rate, or comparison (e.g. "up \
13% sequentially", "margin decreased to 21.5% from 23.6%"), quote that stated figure directly — \
do NOT recompute it yourself from raw numbers. You are prone to arithmetic mistakes; the filing's \
own stated comparison is always more reliable than your mental math. Only compute a new number \
yourself if the question asks for something not already stated anywhere in the context, and even \
then show your work briefly so an error is visible rather than presented as fact.

Keep the answer focused and no longer than necessary."""


def answer_local(question: str) -> QueryResponse:
    store = GraphStore.load()
    if store is None or store.graph.number_of_nodes() == 0:
        return QueryResponse(question=question, mode_used="local",
                              answer="No documents have been ingested yet.", citations=[])
    _, seed_entities = route_question(question)
    seeds, visited, node_lines, facts, citations, raw_chunks = _local_context(question, store, seed_entities)

    context_parts = ["ENTITIES:\n" + "\n".join(node_lines), "\nFACTS:\n" + "\n".join(facts)]
    if raw_chunks:
        context_parts.append("\nSOURCE TEXT:\n" + "\n\n".join(raw_chunks))
    context = "\n".join(context_parts)

    answer = chat(
        config.ANSWER_MODEL, ANSWER_SYSTEM_PROMPT,
        f"Context from knowledge graph:\n{context}\n\nQuestion: {question}",
        max_tokens=800, temperature=0.2,
    )
    return QueryResponse(question=question, mode_used="local", answer=answer,
                          citations=citations, graph_path=visited[:24])


def _rank_communities(question: str, summaries):
    if len(summaries) <= GLOBAL_TOP_N_COMMUNITIES:
        return summaries
    try:
        from .text_embeddings import top_k as embed_top_k
        texts = [f"{s.title}: {s.summary}" for s in summaries]
        hits = embed_top_k(question, texts, k=GLOBAL_TOP_N_COMMUNITIES)
        return [summaries[i] for i, _score in hits]
    except Exception:
        return summaries[:GLOBAL_TOP_N_COMMUNITIES]


def answer_global(question: str) -> QueryResponse:
    summaries = load_summaries()
    if not summaries:
        return QueryResponse(question=question, mode_used="global",
                              answer="No community summaries available yet — run ingestion first.",
                              citations=[])
    ranked = _rank_communities(question, summaries)
    context = "\n\n".join(
        f"[Community: {s.title}] (members: {', '.join(s.members[:8])})\n{s.summary}" for s in ranked
    )
    answer = chat(
        config.ANSWER_MODEL, ANSWER_SYSTEM_PROMPT,
        f"Community summaries:\n{context}\n\nQuestion: {question}",
        max_tokens=900, temperature=0.2,
    )
    citations = [Citation(doc_id=s.title, chunk_id=f"community_{s.community_id}", snippet=s.summary[:300])
                 for s in ranked]
    return QueryResponse(question=question, mode_used="global", answer=answer, citations=citations)


def answer_question(question: str, mode: str = "auto") -> QueryResponse:
    if mode == "auto":
        routed_mode, _ = route_question(question)
    else:
        routed_mode = mode
    if routed_mode == "local":
        return answer_local(question)
    return answer_global(question)