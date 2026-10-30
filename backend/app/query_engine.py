"""
The query engine that makes this "GraphRAG" rather than vector RAG.

For a given question it decides whether to do:

  - LOCAL search (local, multi-hop and consistency questions): find the
    entities the question is about, walk 2 hops of the graph around them,
    PLUS pull in R-GCN-nearest nodes (entities close in embedding space even
    if not directly graph-adjacent — this catches relationships the
    extraction step failed to link explicitly). From that neighborhood, keep
    the facts (relations and per-node values) and source chunks most
    relevant to the question.

  - GLOBAL search (broad, thematic questions): retrieve the community
    summaries most relevant to the question and ground each one with its
    most relevant member chunks, since summaries alone drop most figures.

Local answers are grounded in graph-derived facts and original source chunks.
No vector-database chunks are retrieved by this engine.
"""
from __future__ import annotations
import re
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

from . import config
from .community import load_summaries
from .llm_client import chat_json
from .answering import generate_answer
from .graph_store import GraphStore
from .models import Citation, QueryResponse
from . import rgcn

LOCAL_HOPS = 2
RGCN_EXPANSION_K = 4
GLOBAL_TOP_N_COMMUNITIES = 4
GRAPH_CHUNK_TOP_K = 4
GRAPH_FACT_TOP_K = 40
# A safety bound only: embeddings are cached, so the whole neighborhood is ranked. (A cap of 800
# used to drop facts by discovery order, not relevance, on large EDGAR graphs.)
MAX_CANDIDATE_FACTS = 8000
CHUNKS_PER_COMMUNITY = 2

# Real 12-corpus, 70-question evaluation found GraphRAG's worst losses
# clustered on larger multi-doc graphs, with judge rationales repeatedly
# saying "wrong time period" / "off-topic" -- while vector RAG (pure
# embedding similarity, no entity linking) got the same facts right almost
# every time. Root cause, confirmed with synthetic tests: on a large graph,
# node names differing ONLY by year ("MaxLinear Q2 2025 net revenue" vs
# "MaxLinear Q2 2026 net revenue") score within ~0.04 of each other on raw
# string similarity -- well within noise, so the wrong-period node can win
# seed selection just as easily as the right one. YEAR_PATTERN below is
# used to penalize candidates whose year conflicts with the year explicitly
# named in the search term, since real-world name variation can flip a
# ~0.04 gap in either direction.
YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")
YEAR_CONFLICT_PENALTY = 0.4  # multiplicative penalty applied to a candidate's score


def _years_in(text: str) -> set:
    return set(YEAR_PATTERN.findall(text))


ROUTER_SYSTEM_PROMPT = """Classify a question about a set of financial reports/filings into \
exactly one of these categories -- the SAME taxonomy this system's own benchmark generator uses \
(see generate_benchmark.py), so classification here and question labeling there stay consistent:

- "local": the question is about specific named companies/metrics/periods and their direct \
  relationships or values (e.g. "What was MaxLinear's Q2 2025 net revenue?", "Who was appointed \
  Five9's CFO?").
- "global": requires synthesizing across many entities/documents or asks for a broad theme, \
  comparison, or summary (e.g. "Compare revenue growth across all three companies", "Which \
  companies improved margins this quarter?", "Summarize the key risks across all filings.").
- "multi_hop": requires connecting two or more specific facts (may be within one document or \
  across documents) to answer, e.g. comparing two metrics or two periods.
- "conflict": asks whether something is consistent or inconsistent across the documents (a \
  definition, a period boundary, a reported figure).

Also extract up to 4 key entity names mentioned or implied in the question (company names, \
metric names, periods) for local search seeding, regardless of category.

Return ONLY JSON, no preamble: {"category": "local"|"global"|"multi_hop"|"conflict", \
"entities": [str, ...]}
"""


def classify_query(question: str) -> Tuple[str, List[str]]:
    """Full 4-way classification, matching BenchmarkQuestion's category
    taxonomy exactly (local/global/multi_hop/conflict). This single call
    serves two consumers: (1) route_question() below collapses it to
    local/global for GraphRAG's own internal search-strategy dispatch,
    same behavior as before this edit; (2) the engine router (which
    decides GraphRAG vs vector RAG -- not built in this file, see
    corpus_router.py) can use the full 4-way category as a feature,
    without needing a second LLM call at query time."""
    data = chat_json(config.QUERY_MODEL, ROUTER_SYSTEM_PROMPT, question, max_tokens=250)
    category = data.get("category", "global")
    if category not in ("local", "global", "multi_hop", "conflict"):
        category = "global"
    return category, data.get("entities", []) or []


def route_question(question: str) -> Tuple[str, List[str]]:
    """Local/global search mode and seed entities for one question. Only "global" questions use
    global search: multi-hop and consistency questions name specific entities whose neighborhood
    holds the facts to connect, and community summaries alone drop most figures."""
    category, entities = classify_query(question)
    return ("global" if category == "global" else "local"), entities


def _year_aware_score(name: str, candidate: str) -> float:
    """
    String similarity with a penalty for year conflicts. Plain
    SequenceMatcher.ratio() barely distinguishes "Q2 2025 net revenue" from
    "Q2 2026 net revenue" (a ~0.04 gap in testing) since only 4 characters
    differ out of a long string -- on a large graph with many near-identical
    period-labeled entities, that gap is well within noise and the wrong
    period can win. If the search name specifies a year and the candidate
    specifies a DIFFERENT year, that's an unambiguous signal the candidate
    is wrong regardless of how similar the rest of the string looks.
    """
    base = SequenceMatcher(None, name.lower(), candidate.lower()).ratio()
    name_years = _years_in(name)
    cand_years = _years_in(candidate)
    if name_years and cand_years and name_years.isdisjoint(cand_years):
        return base * YEAR_CONFLICT_PENALTY
    return base


def _fuzzy_match_nodes(names: List[str], store: GraphStore, top_n: int = 4) -> List[str]:
    all_nodes = list(store.graph.nodes())
    matched = []
    for name in names:
        scored = sorted(
            all_nodes, key=lambda n: _year_aware_score(name, n),
            reverse=True,
        )
        best = scored[0] if scored else None
        best_score = _year_aware_score(name, best) if best else 0.0
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


def _rank(question: str, texts: List[str], k: int) -> List[int]:
    """Indices of the k texts most similar to the question (local embeddings), in rank order.
    Falls back to discovery order if the embedding backend is unavailable."""
    if len(texts) <= 1:
        return list(range(len(texts)))
    try:
        from .text_embeddings import top_k as embed_top_k
        return [i for i, _score in embed_top_k(question, texts, k=min(k, len(texts)))]
    except Exception:
        return list(range(min(k, len(texts))))


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


def _candidate_facts(store: GraphStore, nodes: set) -> List[Tuple[str, str, str, str]]:
    """(fact text, chunk_id, doc_id, node) for edges inside the neighborhood and value facts on its nodes."""
    facts = []
    for u, v, data in store.graph.edges(data=True):
        if u in nodes and v in nodes:
            facts.append((f"- {u} --[{data.get('predicate')}]--> {v}  (source: {data.get('doc_id')}; {data.get('evidence') or ''})",
                          data.get("chunk_id"), data.get("doc_id"), u))
    for n in sorted(nodes):
        if store.graph.has_node(n):
            for value in store.graph.nodes[n].get("values", []):
                facts.append((f"- {n} | {value['predicate']}: {value['value']}  (source: {value['doc_id']}; {value.get('evidence') or ''})",
                              value.get("chunk_id"), value.get("doc_id"), n))
    return facts


def warm_embeddings(store: GraphStore) -> int:
    """Embed every fact and chunk once after ingestion, as vector RAG indexes its chunks, so
    query latency measures search rather than first-use embedding. Returns the text count."""
    from .text_embeddings import embed_texts
    texts = [f[0] for f in _candidate_facts(store, set(store.graph.nodes()))]
    texts += [c["text"] for c in store.chunks.values() if c.get("text")]
    for start in range(0, len(texts), 512):
        embed_texts(texts[start:start + 512])
    return len(texts)


def _local_context(question: str, store: GraphStore, seed_entities: List[str]):
    """Neighborhood of the question's entities: graph hops plus R-GCN neighbors, then the most
    question-relevant facts and grounding chunks. Ranking matters because a company node links
    every metric it reports, so two hops can reach most of the graph."""
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

    candidates = _candidate_facts(store, visited)[:MAX_CANDIDATE_FACTS]
    ranked = [candidates[i] for i in _rank(question, [f[0] for f in candidates], GRAPH_FACT_TOP_K)]
    facts = [f[0] for f in ranked]

    # Grounding chunks: those cited by the top facts plus the neighborhood's other chunks,
    # ordered by relevance to the question.
    cited = list(dict.fromkeys(f[1] for f in ranked if f[1] in store.chunks))
    others = sorted({c for n in visited if store.graph.has_node(n) for c in store.chunk_ids_of(n)} - set(cited))
    chunk_ids = cited + others
    order = _rank(question, [store.chunks[c]["text"] for c in chunk_ids], GRAPH_CHUNK_TOP_K)
    top_chunks = [chunk_ids[i] for i in order]

    citations: List[Citation] = []
    for cid in top_chunks + [c for c in cited if c not in top_chunks]:
        chunk = store.chunks.get(cid, {})
        citations.append(Citation(doc_id=chunk.get("doc_id", ""), chunk_id=cid, snippet=chunk.get("text", "")[:300]))

    fact_nodes = {f[3] for f in ranked} | set(seeds)
    node_lines = []
    for n in sorted(fact_nodes):
        if store.graph.has_node(n):
            d = store.graph.nodes[n]
            tag = " [via R-GCN]" if n in rgcn_neighbors else ""
            node_lines.append(f"- {n} ({d.get('type')}){tag}: {d.get('description', '')}")

    raw_chunks = [f"[{store.chunks[c].get('doc_id')} | {c}]\n{store.chunks[c]['text']}"
                  for c in top_chunks if store.chunks.get(c, {}).get("text")]
    return seeds, list(visited), node_lines, facts, citations[:14], raw_chunks


def _bounded(parts: List[str]) -> str:
    """Join context sections, trimming later ones to the graph context budget."""
    out, used = [], 0
    for part in parts:
        room = config.GRAPH_CONTEXT_CHARS - used
        if room <= 200:
            break
        out.append(part[:room])
        used += min(len(part), room)
    return "\n".join(out)


def answer_local(question: str, entities: List[str]) -> Optional[QueryResponse]:
    """Local search; None when no part of the graph matches the question's entities."""
    store = GraphStore.load()
    if store is None or store.graph.number_of_nodes() == 0:
        return QueryResponse(question=question, mode_used="local",
                              answer="No documents have been ingested yet.", citations=[])
    seeds, visited, node_lines, facts, citations, raw_chunks = _local_context(question, store, entities)
    if not seeds or not facts:
        return None
    context = _bounded(["FACTS:\n" + "\n".join(facts),
                        ("\nSOURCE TEXT:\n" + "\n\n".join(raw_chunks)) if raw_chunks else "",
                        "\nENTITIES:\n" + "\n".join(node_lines)])
    answer = generate_answer(question, context)
    return QueryResponse(question=question, mode_used="local", answer=answer,
                          citations=citations, graph_path=visited[:24])


def _rank_communities(question: str, summaries):
    if len(summaries) <= GLOBAL_TOP_N_COMMUNITIES:
        return summaries
    order = _rank(question, [f"{s.title}: {s.summary}" for s in summaries], GLOBAL_TOP_N_COMMUNITIES)
    return [summaries[i] for i in order]


def answer_global(question: str) -> QueryResponse:
    """Global search: the most relevant community summaries for breadth, each grounded by its
    most relevant member chunks. Summaries alone lose figures (they kept 42% of the numbers
    FinQA questions needed), so the answer model also sees the source text they summarize."""
    summaries = load_summaries()
    if not summaries:
        return QueryResponse(question=question, mode_used="global",
                              answer="No community summaries available yet — run ingestion first.",
                              citations=[])
    ranked = _rank_communities(question, summaries)
    store = GraphStore.load()

    def members(summary):
        return [m for m in summary.members if store is not None and store.graph.has_node(m)]

    def source_docs(summary):
        return sorted({doc for m in members(summary) for doc in store.graph.nodes[m].get("source_docs", [])})

    blocks, grounding, citations, used_chunks = [], [], [], set()
    for s in ranked:
        blocks.append(f"[Community: {s.title}; source documents: {', '.join(source_docs(s))}]\n{s.summary}")
        if store is None or s.level > 0:
            continue  # the corpus-level rollup has no single neighborhood to ground
        chunk_ids = sorted({c for m in members(s) for c in store.chunk_ids_of(m)} - used_chunks)
        for i in _rank(question, [store.chunks[c]["text"] for c in chunk_ids], CHUNKS_PER_COMMUNITY):
            cid = chunk_ids[i]
            used_chunks.add(cid)
            chunk = store.chunks[cid]
            grounding.append(f"[{chunk.get('doc_id')} | {cid}]\n{chunk['text']}")
            citations.append(Citation(doc_id=chunk.get("doc_id", ""), chunk_id=cid, snippet=chunk["text"][:300]))
    context = _bounded(["COMMUNITY SUMMARIES:\n" + "\n\n".join(blocks),
                        ("\nSOURCE TEXT:\n" + "\n\n".join(grounding)) if grounding else ""])
    answer = generate_answer(question, context)
    citations += [Citation(doc_id=doc, chunk_id=f"community_{s.community_id}", snippet=s.summary[:300])
                  for s in ranked for doc in source_docs(s)]
    return QueryResponse(question=question, mode_used="global", answer=answer, citations=citations)


def answer_question(question: str, mode: str = "auto", classification: Optional[Tuple[str, List[str]]] = None) -> QueryResponse:
    """Answer from graph neighborhoods (local) or community summaries plus their sources (global).

    classification is (category, entities) from classify_query when the caller already has it
    (the router does), so a query costs one classification call instead of two. Only "global"
    questions use global search: multi-hop and consistency questions are about specific
    entities, whose neighborhood holds the facts to connect or compare."""
    category, entities = classification if classification is not None else classify_query(question)
    routed_mode = mode if mode in ("local", "global") else ("global" if category == "global" else "local")
    if routed_mode == "local":
        response = answer_local(question, entities)
        if response is not None:
            return response
    return answer_global(question)
