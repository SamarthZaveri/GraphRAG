"""Dependency-free BM25 and reciprocal-rank fusion for financial text."""
import math
import re
from collections import Counter


def tokenize(text):
    # Keep years, decimals, and metric names; normalize punctuation/case.
    return re.findall(r"[a-z]+|\d+(?:\.\d+)?", text.lower())


def bm25_rank(question, documents, limit=20, k1=1.5, b=0.75):
    if not documents or limit <= 0:
        return []
    terms = set(tokenize(question))
    if not terms:
        return []
    counts = [Counter(tokenize(doc)) for doc in documents]
    lengths = [sum(c.values()) for c in counts]
    average = sum(lengths) / len(lengths) or 1.0
    df = {term: sum(term in c for c in counts) for term in terms}
    scores = []
    for i, c in enumerate(counts):
        score = 0.0
        for term in terms:
            frequency = c[term]
            if frequency:
                idf = math.log(1 + (len(counts) - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * frequency * (k1 + 1) / (frequency + k1 * (1 - b + b * lengths[i] / average))
        if score > 0:
            scores.append((i, score))
    return sorted(scores, key=lambda item: (-item[1], item[0]))[:limit]


def reciprocal_rank_fusion(rankings, limit=6, k=60):
    if k <= 0:
        raise ValueError("RRF k must be positive")
    scores = {}
    for ranking in rankings:
        seen = set()
        for rank, chunk_id in enumerate(ranking, 1):
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)
    return sorted(scores, key=lambda cid: (-scores[cid], cid))[:limit]
