"""
Shared local text-embedding helper.

Used in three places to make retrieval more cohesive instead of relying on
pure string matching everywhere:
  1. Entity resolution (graph_store.py) — tier-2 dedup for names that don't
     string-match but describe the same thing.
  2. Query-to-node linking (query_engine.py) — when the router's extracted
     entity names don't exactly match a graph node, find the closest node by
     meaning instead of substring matching alone.
  3. Global-search community ranking (query_engine.py) — pick the community
     summaries actually relevant to the question instead of dumping all of
     them into the prompt.

Wraps Chroma's default local sentence-embedding function (all-MiniLM-L6-v2
via ONNX), which is already a dependency for the vector-RAG baseline, so
this adds no new package and no new network dependency beyond the one-time
model download Chroma already does.
"""
from __future__ import annotations
from typing import List

import numpy as np

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        _embedder = DefaultEmbeddingFunction()
    return _embedder


def embed_texts(texts: List[str]) -> np.ndarray:
    """Returns an (n, dim) float32 array. Empty input -> shape (0, 0)."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    embedder = get_embedder()
    vecs = embedder(texts)
    return np.asarray(vecs, dtype=np.float32)


def cosine_sim_matrix(query_vecs: np.ndarray, corpus_vecs: np.ndarray) -> np.ndarray:
    """Returns an (n_query, n_corpus) cosine-similarity matrix."""
    if query_vecs.size == 0 or corpus_vecs.size == 0:
        return np.zeros((query_vecs.shape[0], corpus_vecs.shape[0]))
    qn = query_vecs / (np.linalg.norm(query_vecs, axis=1, keepdims=True) + 1e-8)
    cn = corpus_vecs / (np.linalg.norm(corpus_vecs, axis=1, keepdims=True) + 1e-8)
    return qn @ cn.T


def top_k(query_text: str, corpus_texts: List[str], k: int = 5) -> List[tuple]:
    """Returns [(index, score), ...] for the top-k most similar corpus_texts
    to query_text, sorted by descending score."""
    if not corpus_texts:
        return []
    vecs = embed_texts([query_text] + corpus_texts)
    query_vec, corpus_vecs = vecs[:1], vecs[1:]
    sims = cosine_sim_matrix(query_vec, corpus_vecs)[0]
    order = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in order]