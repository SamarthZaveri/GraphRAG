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

from . import config, workspace
from .models import ExtractionResult, Triple

def _path(name):
    return workspace.state_dir() / name

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


_CORPORATE_SUFFIX = re.compile(r",?\s+(inc|corp|corporation|co|company|llc|plc|ltd|holdings)\.?$", re.I)


def _match_form(name: str) -> str:
    """Comparison form: lowercase without a trailing corporate suffix ("Delta Air Lines, Inc.")."""
    return _CORPORATE_SUFFIX.sub("", name.strip()).lower()


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, _match_form(a), _match_form(b)).ratio()


_NUMBER_TOKEN = re.compile(r"\d+(?:\.\d+)?|\b[qh][1-4]\b|\bfy\d*\b", re.I)


def _numbers(name: str) -> set:
    return {t.lower() for t in _NUMBER_TOKEN.findall(name)}


def _mergeable(a: str, b: str) -> bool:
    """Names that differ only by year, quarter or amount are different entities: "net revenue
    Q2 2025" vs "net revenue Q2 2024" are 0.96 similar as strings, yet merging them would
    attach one period's facts to the other."""
    return _numbers(a) == _numbers(b)


# Objects that are values, not things: figures, percentages, dated amounts and direction words.
_LITERAL_WORDS = {"up", "down", "increase", "decrease", "increased", "decreased", "flat", "higher", "lower",
                  "unchanged", "yes", "no", "true", "false", "none", "n/a"}


def is_literal(text: str) -> bool:
    text = text.strip().lower()
    if text in _LITERAL_WORDS:
        return True
    return bool(re.search(r"\d", text)) and len(text.split()) <= 6 and not re.search(r"\b(inc|corp|llc|plc|ltd)\b", text)


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
        self.exact_only: set = set()  # deterministic table names: exact matches only
        self._embeddings_dirty = True  # set True if canonical_embeddings needs (re)building

    @staticmethod
    def _text_repr(name: str, type_: str, description: str) -> str:
        return f"{name} ({type_}): {description or ''}".strip()

    def rebuild_embeddings(self):
        """(Re)computes embeddings for every canonical entity that's missing one.
        Called after loading resolver state from disk, and lazily during resolve()."""
        missing = [c for c in self.canonical_names if c not in self.canonical_embeddings and c not in self.exact_only]
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
            from .llm_client import chat_json
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

    def resolve(self, raw_name: str, type_: str = "Other", description: str = "", fuzzy: bool = True) -> str:
        """fuzzy=False registers unseen names without fuzzy or embedding matching: for
        deterministically generated table names, where an exact match is the only correct one."""
        norm = _normalize(raw_name)
        key = norm.lower()
        if key in self._lookup:
            return self._lookup[key]
        if not fuzzy:
            return self._register(norm, key, type_, description, embed=False)

        # Tier 1: string fuzzy match
        best, best_score = None, 0.0
        for canon in self.canonical_names:
            if not _mergeable(norm, canon):
                continue
            matcher = SequenceMatcher(None, _match_form(norm), _match_form(canon))
            # Cheap upper bounds first: most pairs cannot reach the merge threshold.
            if matcher.real_quick_ratio() < self.string_threshold or matcher.quick_ratio() < self.string_threshold:
                continue
            score = matcher.ratio()
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
                          and c in self.canonical_embeddings and _mergeable(norm, c)]
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

        return self._register(norm, key, type_, description)

    def _register(self, norm, key, type_, description, embed=True):
        """No merge: a new canonical entity. Unembedded (table) names never become Tier-2 merge
        targets, so a narrative alias cannot be fuzzily attached to a specific table row."""
        self.canonical_names.append(norm)
        self.canonical_types[norm] = type_
        self.canonical_descriptions[norm] = description or ""
        if not embed:
            self.exact_only.add(norm)
        else:
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
            declared = {}
            # Table metric names are deterministic, so only exact matches are correct; the table's
            # company still resolves fuzzily, linking table facts to the narrative's "Delta Air Lines".
            table = res.chunk_id.endswith("::tables")
            for ent in res.entities:
                canon = self.resolver.resolve(ent.name, ent.type, ent.description or "",
                                              fuzzy=not table or ent.type == "Company")
                declared[ent.name.strip().lower()] = ent.type
                self._add_node(canon, ent.type, ent.description or "", res.doc_id)
            for tr in res.triples:
                subject_type = declared.get(tr.subject.strip().lower(), "Other")
                s = self.resolver.resolve(tr.subject, subject_type, fuzzy=not table or subject_type == "Company")
                self._add_node(s, subject_type, "", tr.doc_id)
                if declared.get(tr.object.strip().lower()) not in ("Company", "Segment", "Person") and is_literal(tr.object):
                    # A value is a fact about its subject. As a shared node, "$100.00" or "2003"
                    # would link every unrelated metric that happens to share it.
                    self.graph.nodes[s].setdefault("values", []).append({
                        "predicate": tr.predicate, "value": tr.object.strip(), "doc_id": tr.doc_id,
                        "chunk_id": tr.chunk_id, "evidence": tr.evidence})
                    continue
                o = self.resolver.resolve(tr.object, declared.get(tr.object.strip().lower(), "Other"))
                self._add_node(o, declared.get(tr.object.strip().lower(), "Other"), "", tr.doc_id)
                self.graph.add_edge(
                    s, o, predicate=tr.predicate, doc_id=tr.doc_id,
                    chunk_id=tr.chunk_id, evidence=tr.evidence,
                )

    def _add_node(self, name, type_, description, doc_id):
        if self.graph.has_node(name):
            node = self.graph.nodes[name]
            node["source_docs"] = sorted(set(node.get("source_docs", [])) | {doc_id})
            if description and len(description) > len(node.get("description") or ""):
                node["description"] = description
            if node.get("type") in (None, "Other") and type_ != "Other":
                node["type"] = type_
        else:
            self.graph.add_node(name, type=type_, description=description, source_docs=[doc_id], values=[])

    def chunk_ids_of(self, node) -> set:
        """Chunks that ground a node: those cited by its edges and by its value facts."""
        data = self.graph.nodes[node]
        ids = {v["chunk_id"] for v in data.get("values", []) if v.get("chunk_id")}
        for _, _, edge in self.graph.in_edges(node, data=True):
            ids.add(edge.get("chunk_id"))
        for _, _, edge in self.graph.out_edges(node, data=True):
            ids.add(edge.get("chunk_id"))
        return {i for i in ids if i in self.chunks}

    def stats(self) -> dict:
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
        }

    def save(self):
        with open(_path("graph.gpickle"), "wb") as f:
            pickle.dump(self.graph, f)
        with open(_path("chunks.json"), "w") as f:
            json.dump(self.chunks, f)
        with open(_path("meta.json"), "w") as f:
            json.dump({
                "canonical_names": self.resolver.canonical_names,
                "canonical_types": self.resolver.canonical_types,
                "canonical_descriptions": self.resolver.canonical_descriptions,
                "lookup": self.resolver._lookup,
                "exact_only": sorted(self.resolver.exact_only),
            }, f)

    @classmethod
    def load(cls) -> Optional["GraphStore"]:
        if not _path("graph.gpickle").exists():
            return None
        with open(_path("graph.gpickle"), "rb") as f:
            graph = pickle.load(f)
        chunks = {}
        if _path("chunks.json").exists():
            with open(_path("chunks.json")) as f:
                chunks = json.load(f)
        resolver = EntityResolver()
        if _path("meta.json").exists():
            with open(_path("meta.json")) as f:
                meta = json.load(f)
            resolver.canonical_names = meta.get("canonical_names", [])
            resolver.canonical_types = meta.get("canonical_types", {})
            resolver.canonical_descriptions = meta.get("canonical_descriptions", {})
            resolver._lookup = meta.get("lookup", {})
            resolver.exact_only = set(meta.get("exact_only", []))
        # embeddings are rebuilt lazily on first resolve() call in this session
        return cls(graph=graph, chunks=chunks, resolver=resolver)


_stores: dict = {}  # one cached store per workspace (None = shared corpus)


def get_store(fresh: bool = False) -> GraphStore:
    ws = workspace.current()
    if fresh or ws not in _stores:
        loaded = None if fresh else GraphStore.load()
        _stores[ws] = loaded if loaded is not None else GraphStore()
    return _stores[ws]


def reset_store():
    store = _stores[workspace.current()] = GraphStore()
    return store