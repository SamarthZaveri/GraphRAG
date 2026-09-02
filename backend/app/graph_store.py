"""
Builds and persists the NetworkX knowledge graph from extracted triples.

Entity resolution is tiered (Phase 2 item from the PRD — real dedup, not
just an MVP stopgap):

  Tier 1 — exact/fuzzy string match on the normalized name (fast, high
  precision; catches "MaxLinear, Inc." vs "MaxLinear Inc").

  Tier 2 — local sentence-embedding similarity between "name + type +
  description" of the new entity and existing same-type canonical entities
  (catches aliases string matching misses, e.g. "the Company" vs the
  company's full name, when their descriptions overlap). Above a high
  threshold it auto-merges; in a middle band it asks the local LLM once to
  confirm the two descriptions refer to the same real-world entity before
  merging, so a borderline embedding match can't silently over-merge two
  different companies with similar descriptions.

This trades a little ingestion latency for real dedup quality, and degrades
gracefully to Tier-1-only if the local embedding backend can't be reached.
"""
from __future__ import annotations
import json
import pickle
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional

import networkx as nx
import numpy as np

from . import config
from .models import ExtractionResult, Triple

GRAPH_PATH = config.GRAPH_STATE_DIR / "graph.gpickle"
CHUNKS_PATH = config.GRAPH_STATE_DIR / "chunks.json"
META_PATH = config.GRAPH_STATE_DIR / "meta.json"

STRING_MERGE_THRESHOLD = 0.92
EMBED_AUTO_MERGE_THRESHOLD = 0.93
EMBED_CANDIDATE_THRESHOLD = 0.80

MERGE_CONFIRM_SYSTEM_PROMPT = """Two entity descriptions were extracted from financial \
documents and look like they *might* refer to the same real-world entity (company, metric, \
person, etc.). Decide whether they actually are the same entity.

Answer conservatively: only say they're the same if you're confident (e.g. "MaxLinear, Inc."
and "MaxLinear" referring to the same company, or "Q2 2025" and "the second quarter of 2025"
referring to the same period). Different companies, different metrics, or different fiscal
periods are NOT the same even if superficially similar.

Return ONLY JSON, no preamble: {"same_entity": true|false}
"""


def _normalize(name: str) -> str:
    name = name.strip()
    name = re.sub(r"^(the|this)\s+", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+", " ", name)
    return name


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


class EntityResolver:
    """Tiered string + embedding (+ LLM-confirmed) entity resolution."""

    def __init__(self, string_threshold: float = STRING_MERGE_THRESHOLD,
                 embed_auto_threshold: float = EMBED_AUTO_MERGE_THRESHOLD,
                 embed_candidate_threshold: float = EMBED_CANDIDATE_THRESHOLD):
        self.string_threshold = string_threshold
        self.embed_auto_threshold = embed_auto_threshold
        self.embed_candidate_threshold = embed_candidate_threshold

        self.canonical_names: List[str] = []
        self.canonical_types: Dict[str, str] = {}
        self.canonical_descriptions: Dict[str, str] = {}
        self.canonical_embeddings: Dict[str, np.ndarray] = {}
        self._lookup: Dict[str, str] = {}
        self._llm_confirm_cache: Dict[str, bool] = {}
        self._embeddings_dirty = True  # set True if canonical_embeddings needs (re)building

    @staticmethod
    def _text_repr(name: str, type_: str, description: str) -> str:
        return f"{name} ({type_}): {description or ''}".strip()

    def rebuild_embeddings(self):
        """(Re)computes embeddings for every canonical entity that's missing one.
        Called after loading resolver state from disk, and lazily during resolve()."""
        missing = [c for c in self.canonical_names if c not in self.canonical_embeddings]
        if not missing:
            self._embeddings_dirty = False
            return
        try:
            from .text_embeddings import embed_texts
            texts = [self._text_repr(c, self.canonical_types.get(c, "Other"),
                                      self.canonical_descriptions.get(c, "")) for c in missing]
            vecs = embed_texts(texts)
            for name, vec in zip(missing, vecs):
                self.canonical_embeddings[name] = vec
        except Exception:
            pass  # embedding backend unavailable — Tier 2 just won't fire this session
        self._embeddings_dirty = False

    def _llm_confirm_merge(self, name_a: str, desc_a: str, name_b: str, desc_b: str) -> bool:
        cache_key = f"{name_a.lower()}||{name_b.lower()}"
        if cache_key in self._llm_confirm_cache:
            return self._llm_confirm_cache[cache_key]
        try:
            from .ollama_client import chat_json
            data = chat_json(
                config.EXTRACTION_MODEL, MERGE_CONFIRM_SYSTEM_PROMPT,
                f"Entity A: {name_a} — {desc_a}\nEntity B: {name_b} — {desc_b}",
                max_tokens=60,
            )
            result = bool(data.get("same_entity", False))
        except Exception:
            result = False  # can't reach the LLM — be conservative, don't merge
        self._llm_confirm_cache[cache_key] = result
        return result

    def resolve(self, raw_name: str, type_: str = "Other", description: str = "") -> str:
        norm = _normalize(raw_name)
        key = norm.lower()
        if key in self._lookup:
            return self._lookup[key]

        # Tier 1: string fuzzy match
        best, best_score = None, 0.0
        for canon in self.canonical_names:
            score = _similar(norm, canon)
            if score > best_score:
                best, best_score = canon, score
        if best is not None and best_score >= self.string_threshold:
            self._lookup[key] = best
            return best

        # Tier 2: embedding similarity among same-type canonical entities
        try:
            if self._embeddings_dirty:
                self.rebuild_embeddings()
            candidates = [c for c in self.canonical_names if self.canonical_types.get(c) == type_
                          and c in self.canonical_embeddings]
            if candidates:
                from .text_embeddings import embed_texts, cosine_sim_matrix
                query_vec = embed_texts([self._text_repr(norm, type_, description)])
                cand_vecs = np.stack([self.canonical_embeddings[c] for c in candidates])
                sims = cosine_sim_matrix(query_vec, cand_vecs)[0]
                top_i = int(np.argmax(sims))
                top_score, top_name = float(sims[top_i]), candidates[top_i]
                if top_score >= self.embed_auto_threshold:
                    self._lookup[key] = top_name
                    return top_name
                if top_score >= self.embed_candidate_threshold:
                    if self._llm_confirm_merge(norm, description, top_name,
                                                self.canonical_descriptions.get(top_name, "")):
                        self._lookup[key] = top_name
                        return top_name
        except Exception:
            pass  # embedding backend unavailable — fall through to Tier 1 result (new entity)

        # No merge: register as a new canonical entity
        self.canonical_names.append(norm)
        self.canonical_types[norm] = type_
        self.canonical_descriptions[norm] = description or ""
        try:
            from .text_embeddings import embed_texts
            self.canonical_embeddings[norm] = embed_texts([self._text_repr(norm, type_, description)])[0]
        except Exception:
            pass
        self._lookup[key] = norm
        return norm


@dataclass
class GraphStore:
    graph: nx.MultiDiGraph = field(default_factory=nx.MultiDiGraph)
    chunks: Dict[str, dict] = field(default_factory=dict)  # chunk_id -> {doc_id, text}
    resolver: EntityResolver = field(default_factory=EntityResolver)

    def ingested_doc_ids(self) -> set:
        return {d for _, data in self.graph.nodes(data=True) for d in data.get("source_docs", [])}

    def ingest_document_chunks(self, doc_id: str, chunk_texts: List[dict],
                                extraction_results: List[ExtractionResult]):
        for c in chunk_texts:
            self.chunks[c["chunk_id"]] = {"doc_id": c["doc_id"], "text": c["text"]}

        for res in extraction_results:
            for ent in res.entities:
                canon = self.resolver.resolve(ent.name, ent.type, ent.description or "")
                if self.graph.has_node(canon):
                    node = self.graph.nodes[canon]
                    node["source_docs"] = sorted(set(node.get("source_docs", [])) | {res.doc_id})
                    if ent.description and len(ent.description) > len(node.get("description") or ""):
                        node["description"] = ent.description
                else:
                    self.graph.add_node(
                        canon, type=ent.type, description=ent.description or "",
                        source_docs=[res.doc_id],
                    )
            for tr in res.triples:
                s = self.resolver.resolve(tr.subject)
                o = self.resolver.resolve(tr.object)
                if not self.graph.has_node(s):
                    self.graph.add_node(s, type="Other", description="", source_docs=[tr.doc_id])
                if not self.graph.has_node(o):
                    self.graph.add_node(o, type="Other", description="", source_docs=[tr.doc_id])
                self.graph.add_edge(
                    s, o, predicate=tr.predicate, doc_id=tr.doc_id,
                    chunk_id=tr.chunk_id, evidence=tr.evidence,
                )

    def stats(self) -> dict:
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
        }

    def save(self):
        config.GRAPH_STATE_DIR.mkdir(parents=True, exist_ok=True)
        with open(GRAPH_PATH, "wb") as f:
            pickle.dump(self.graph, f)
        with open(CHUNKS_PATH, "w") as f:
            json.dump(self.chunks, f)
        with open(META_PATH, "w") as f:
            json.dump({
                "canonical_names": self.resolver.canonical_names,
                "canonical_types": self.resolver.canonical_types,
                "canonical_descriptions": self.resolver.canonical_descriptions,
                "lookup": self.resolver._lookup,
            }, f)

    @classmethod
    def load(cls) -> Optional["GraphStore"]:
        if not GRAPH_PATH.exists():
            return None
        with open(GRAPH_PATH, "rb") as f:
            graph = pickle.load(f)
        chunks = {}
        if CHUNKS_PATH.exists():
            with open(CHUNKS_PATH) as f:
                chunks = json.load(f)
        resolver = EntityResolver()
        if META_PATH.exists():
            with open(META_PATH) as f:
                meta = json.load(f)
            resolver.canonical_names = meta.get("canonical_names", [])
            resolver.canonical_types = meta.get("canonical_types", {})
            resolver.canonical_descriptions = meta.get("canonical_descriptions", {})
            resolver._lookup = meta.get("lookup", {})
        # embeddings are rebuilt lazily on first resolve() call in this session
        return cls(graph=graph, chunks=chunks, resolver=resolver)


_store: Optional[GraphStore] = None


def get_store(fresh: bool = False) -> GraphStore:
    global _store
    if fresh or _store is None:
        loaded = None if fresh else GraphStore.load()
        _store = loaded if loaded is not None else GraphStore()
    return _store


def reset_store():
    global _store
    _store = GraphStore()
    return _store