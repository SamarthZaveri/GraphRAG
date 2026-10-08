"""BM25 + dense vector retrieval over the baseline's raw chunks, without a graph."""
from . import config, vector_baseline
from .answering import generate_answer
from .lexical import bm25_rank, reciprocal_rank_fusion
from .models import Citation, QueryResponse


def retrieve(question, top_k=config.HYBRID_TOP_K):
    collection = vector_baseline.get_collection()
    if collection.count() == 0:
        return []
    # Read the current collection so replace/add ingestion cannot leave a stale BM25 index.
    # For this experimental corpus size rebuilding BM25 is cheap; large deployments
    # should replace this scan with a persistent lexical index.
    data = collection.get(include=["documents", "metadatas"])
    records = {cid: {"chunk_id": cid, "text": text, "doc_id": (meta or {}).get("doc_id", "")}
               for cid, text, meta in zip(data["ids"], data["documents"], data["metadatas"])}
    ids = list(records)
    lexical = [ids[i] for i, _ in bm25_rank(question, [records[cid]["text"] for cid in ids], config.HYBRID_CANDIDATES)]
    dense = collection.query(query_texts=[question], n_results=min(config.HYBRID_CANDIDATES, len(ids)))
    fused = reciprocal_rank_fusion([lexical, dense["ids"][0]], top_k, config.RRF_K)
    return [records[cid] for cid in fused if cid in records]


def answer_question(question, top_k=config.HYBRID_TOP_K):
    chunks = retrieve(question, top_k)
    if not chunks:
        return QueryResponse(question=question, mode_used="local", engine_used="hybrid_rag",
                             answer="No documents have been ingested into the hybrid index.", citations=[])
    context = "\n\n".join(f"[{c['chunk_id']} | doc_id={c['doc_id']}]\n{c['text']}" for c in chunks)
    return QueryResponse(question=question, mode_used="local", engine_used="hybrid_rag",
                         answer=generate_answer(question, context),
                         citations=[Citation(doc_id=c["doc_id"], chunk_id=c["chunk_id"], snippet=c["text"][:300]) for c in chunks])
