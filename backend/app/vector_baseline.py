"""
Vanilla vector-RAG baseline: chunk -> embed (Chroma's default local
sentence-transformer embedding function) -> similarity search -> stuff
top-k chunks into the prompt. No graph, no entities, no communities. This
is what GraphRAG is benchmarked against.

Note: Chroma's default embedding function downloads a small ONNX model
(all-MiniLM-L6-v2, ~80MB) from Hugging Face the first time it runs, then
caches it locally under ~/.cache/chroma/. After that first run, everything
here is fully offline, same as the Ollama-backed generation.
"""
from __future__ import annotations
from typing import List

import chromadb

from . import config
from .extraction import chunk_text
from .ollama_client import chat
from .models import Citation, QueryResponse

_client = None
_collection = None
COLLECTION_NAME = "ledger_vector_baseline"


def get_collection():
    global _client, _collection
    if _client is None:
        _client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    if _collection is None:
        _collection = _client.get_or_create_collection(COLLECTION_NAME)
    return _collection


def reset_collection():
    global _client, _collection
    if _client is None:
        _client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        _client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    _collection = _client.get_or_create_collection(COLLECTION_NAME)
    return _collection


def ingest_document(doc_id: str, text: str):
    collection = get_collection()
    chunks = chunk_text(text, doc_id)
    if not chunks:
        return
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"doc_id": c["doc_id"]} for c in chunks],
    )


ANSWER_SYSTEM_PROMPT = """You are a plain vector-RAG assistant. Answer the question using ONLY \
the retrieved passages below. If they don't contain the answer, say so. Cite the doc_id for \
each claim."""


def answer_question(question: str, top_k: int = config.VECTOR_TOP_K) -> QueryResponse:
    collection = get_collection()
    if collection.count() == 0:
        return QueryResponse(question=question, mode_used="local",
                              answer="No documents have been ingested into the vector baseline yet.",
                              citations=[])
    results = collection.query(query_texts=[question], n_results=min(top_k, collection.count()))
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    ids = results["ids"][0]

    context = "\n\n".join(
        f"[{ids[i]} | doc_id={metas[i]['doc_id']}]\n{docs[i]}" for i in range(len(docs))
    )
    answer = chat(
        config.ANSWER_MODEL, ANSWER_SYSTEM_PROMPT,
        f"Retrieved passages:\n{context}\n\nQuestion: {question}",
        max_tokens=800, temperature=0.2,
    )
    citations = [Citation(doc_id=metas[i]["doc_id"], chunk_id=ids[i], snippet=docs[i][:300])
                 for i in range(len(docs))]
    return QueryResponse(question=question, mode_used="local", answer=answer, citations=citations)
