"""
Builds and persists the NetworkX knowledge graph from extracted triples.

Entity resolution (MVP stopgap, per PRD risk mitigation): normalize casing/
whitespace and merge near-duplicate names via a simple token-overlap +
substring heuristic (e.g. "Acme Corp" / "Acme Corporation" / "the Company"
alias chains are NOT resolved here — only surface-level string variants are).
Real entity resolution is a Phase 2 item.
"""
from __future__ import annotations
import json
import pickle
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional

import networkx as nx

from . import config
from .models import ExtractionResult, Triple

GRAPH_PATH = config.GRAPH_STATE_DIR / "graph.gpickle"
CHUNKS_PATH = config.GRAPH_STATE_DIR / "chunks.json"
META_PATH = config.GRAPH_STATE_DIR / "meta.json"


def _normalize(name: str) -> str:
    name = name.strip()
    name = re.sub(r"^(the|this)\s+", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+", " ", name)
    return name


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


class EntityResolver:
    """Exact/fuzzy string-matching stopgap for entity resolution."""

    def __init__(self, threshold: float = 0.92):
        self.threshold = threshold
        self.canonical_names: List[str] = []
        self._lookup: Dict[str, str] = {}

    def resolve(self, raw_name: str) -> str:
        norm = _normalize(raw_name)
        key = norm.lower()
        if key in self._lookup:
            return self._lookup[key]
        best, best_score = None, 0.0
        for canon in self.canonical_names:
            score = _similar(norm, canon)
            if score > best_score:
                best, best_score = canon, score
        if best is not None and best_score >= self.threshold:
            self._lookup[key] = best
            return best
        self.canonical_names.append(norm)
        self._lookup[key] = norm
        return norm


@dataclass
class GraphStore:
    graph: nx.MultiDiGraph = field(default_factory=nx.MultiDiGraph)
    chunks: Dict[str, dict] = field(default_factory=dict)  # chunk_id -> {doc_id, text}
    resolver: EntityResolver = field(default_factory=EntityResolver)

    def ingest_document_chunks(self, doc_id: str, chunk_texts: List[dict],
                                extraction_results: List[ExtractionResult]):
        for c in chunk_texts:
            self.chunks[c["chunk_id"]] = {"doc_id": c["doc_id"], "text": c["text"]}

        for res in extraction_results:
            for ent in res.entities:
                canon = self.resolver.resolve(ent.name)
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
            resolver._lookup = meta.get("lookup", {})
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
