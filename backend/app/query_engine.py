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
import re
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
    data = chat_json(config.EXTRACTION_MODEL, ROUTER_SYSTEM_PROMPT, question, max_tokens=250)
    category = data.get("category", "global")
    if category not in ("local", "global", "multi_hop", "conflict"):
        category = "global"
    return category, data.get("entities", []) or []


def route_question(question: str) -> Tuple[str, List[str]]:
    """Backward-compatible local/global mode for GraphRAG's own internal
    search dispatch (see answer_question below) -- unchanged behavior from
    before this edit. Collapses the fuller 4-way category down to what the
    retrieval logic currently understands: "local" stays local search;
    global/multi_hop/conflict all route to global search, since multi_hop
    and conflict both need synthesis across more than one specific spot,
    same as global does."""
    category, entities = classify_query(question)
    mode = "local" if category == "local" else "global"
    return mode, entities


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


def _local_context(question: str, store: GraphStore, seed_entities: List[str], use_hybrid: bool = True):
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

    # Hybrid grounding. ORDER MATTERS here when enabled: vector-similarity
    # chunks go FIRST, graph-traversal chunks go SECOND. Real evaluation
    # data showed GraphRAG's worst failures were "wrong time period"
    # answers on larger graphs -- if a fuzzy-matched seed grabbed the wrong
    # period (see _year_aware_score above), the graph-derived chunk for
    # that wrong period would previously appear FIRST in context, and an
    # LLM instructed to "copy figures exactly as given" can end up copying
    # the wrong period's figure with full confidence. Vector similarity
    # doesn't depend on entity-name string matching, so it's less prone to
    # this specific failure mode -- giving it primacy in context doesn't
    # fix a bad seed match, but it stops a bad seed match from actively
    # out-competing the correct content for the model's attention.
    #
    # use_hybrid=False disables this block entirely, for the graph-only
    # ablation (see RL/run_ablation.py): GraphRAG's own hybrid safety net
    # already pulls in the same vector-similarity chunks vector RAG uses,
    # so a "GraphRAG vs vector RAG" comparison with hybrid always-on isn't
    # really testing graph retrieval against vector retrieval -- it's
    # testing (graph + vector) against (vector alone). Disabling it here
    # isolates what graph structure alone actually contributes.
    raw_chunks = []
    if not use_hybrid:
        for cid in list(chunk_ids_seen)[:HYBRID_CHUNK_TOP_K]:
            chunk = store.chunks.get(cid, {})
            if chunk.get("text"):
                raw_chunks.append(f"[{chunk.get('doc_id')} | {cid}]\n{chunk['text']}")
        return seeds, list(visited), node_lines, facts, citations[:14], raw_chunks

    # Snapshot BEFORE the vector-fallback loop mutates chunk_ids_seen, so
    # the two sources stay cleanly separable.
    graph_derived_chunk_ids = set(chunk_ids_seen)

    try:
        from . import vector_baseline
        collection = vector_baseline.get_collection()
        if collection.count() > 0:
            results = collection.query(query_texts=[question], n_results=min(HYBRID_VECTOR_FALLBACK_K, collection.count()))
            for cid, doc_text, meta in zip(results["ids"][0], results["documents"][0], results["metadatas"][0]):
                if cid in graph_derived_chunk_ids:
                    continue  # will be added in its graph-derived form below; skip the duplicate
                raw_chunks.append(f"[{meta.get('doc_id')} | {cid}]\n{doc_text}")
                chunk_ids_seen.add(cid)
                citations.append(Citation(doc_id=meta.get("doc_id", ""), chunk_id=cid, snippet=doc_text[:300]))
    except Exception:
        pass  # vector baseline not available/ingested yet — graph-only context still works

    # ...then graph-traversal-found chunks (capped), appended AFTER so they
    # don't crowd out the vector-similarity chunks' primacy in context.
    for cid in list(graph_derived_chunk_ids)[:HYBRID_CHUNK_TOP_K]:
        chunk = store.chunks.get(cid, {})
        if chunk.get("text"):
            raw_chunks.append(f"[{chunk.get('doc_id')} | {cid}]\n{chunk['text']}")

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


def answer_local(question: str, use_hybrid: bool = True) -> QueryResponse:
    store = GraphStore.load()
    if store is None or store.graph.number_of_nodes() == 0:
        return QueryResponse(question=question, mode_used="local",
                              answer="No documents have been ingested yet.", citations=[])
    _, seed_entities = route_question(question)
    seeds, visited, node_lines, facts, citations, raw_chunks = _local_context(
        question, store, seed_entities, use_hybrid=use_hybrid
    )

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


def answer_question(question: str, mode: str = "auto", use_hybrid: bool = True) -> QueryResponse:
    """
    use_hybrid only affects LOCAL mode (answer_global uses community
    summaries, a separate mechanism with no vector fallback to toggle).
    Defaults to True, matching production behavior unchanged -- pass False
    for the graph-only ablation (RL/run_ablation.py).
    """
    if mode == "auto":
        routed_mode, _ = route_question(question)
    else:
        routed_mode = mode
    if routed_mode == "local":
        return answer_local(question, use_hybrid=use_hybrid)
    return answer_global(question)