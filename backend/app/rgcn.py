"""
Self-supervised R-GCN node embeddings.

There's no labeled data to train a classifier against (no ground-truth node
classes), so this trains node embeddings the way knowledge-graph embedding
methods do: via link prediction. It learns vectors such that entities
connected by a real edge score higher under a DistMult decoder than
randomly-corrupted negative pairs, refined through relational graph
convolution layers — one weight matrix per relation "bucket" (the extracted
predicates are free text, e.g. "increased sequentially vs", "has value",
so they're bucketed into a small fixed set of relation types first, or the
per-relation weight matrices would never see enough examples of any single
literal predicate string to learn anything).

The resulting embeddings are transductive (defined only for nodes already in
the graph), so they're used for graph-internal retrieval tasks:
  - entity-resolution merge-candidate detection (graph_store.py can compare
    existing nodes to each other)
  - local-search neighborhood expansion (query_engine.py can pull in nodes
    that are close in embedding space even if not directly graph-adjacent —
    this catches relationships the extraction step failed to link explicitly)

Linking an arbitrary NEW query string (e.g. a name the router extracted from
the question) to a graph node is a different problem — that needs a general
text encoder, which is what text_embeddings.py is for. This module doesn't
try to do that.

Entirely optional: if PyTorch isn't installed, every public function here
returns None / a no-op, and the rest of the app works exactly as before.
"""
from __future__ import annotations
import json
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

from . import config

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

EMBEDDINGS_PATH = config.GRAPH_STATE_DIR / "rgcn_embeddings.npy"
NODES_PATH = config.GRAPH_STATE_DIR / "rgcn_nodes.json"

EMBED_DIM = 48
NUM_LAYERS = 2
EPOCHS = 80
LR = 0.02

# Predicates are free text; bucket them into a small fixed relation-type set
# so each relation's weight matrix actually sees enough edges to learn from.
_RELATION_PATTERNS: List[Tuple[str, "re.Pattern"]] = [
    ("has_value", re.compile(r"\bvalue\b|\bwas\b|totaled|amounted|^is\b", re.I)),
    ("in_period", re.compile(r"period|quarter|fiscal|month|year(?!.over.year)", re.I)),
    ("comparison", re.compile(r"increas|decreas|compar|grew|declin|versus|\bvs\b|sequential|"
                               r"year.over.year|yoy|up |down ", re.I)),
    ("reports", re.compile(r"segment|report|division|subsidiary", re.I)),
    ("role", re.compile(r"officer|chairman|ceo|cfo|president|director|appoint", re.I)),
    ("guidance", re.compile(r"expect|guidance|outlook|forecast|project", re.I)),
]
RELATION_NAMES = [name for name, _ in _RELATION_PATTERNS] + ["other"]


def bucket_predicate(predicate: str) -> str:
    for name, pattern in _RELATION_PATTERNS:
        if pattern.search(predicate or ""):
            return name
    return "other"


if TORCH_AVAILABLE:

    class RelGraphConvLayer(nn.Module):
        def __init__(self, in_dim: int, out_dim: int, num_relations: int):
            super().__init__()
            self.rel_weights = nn.ModuleList(
                [nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_relations)]
            )
            self.self_weight = nn.Linear(in_dim, out_dim, bias=False)

        def forward(self, x: "torch.Tensor", edges_by_rel: Dict[int, Tuple["torch.Tensor", "torch.Tensor"]]):
            out = self.self_weight(x)
            for rel_id, (src, dst) in edges_by_rel.items():
                if src.numel() == 0:
                    continue
                msg = self.rel_weights[rel_id](x[src])
                # mean aggregation per destination node (simple degree norm)
                out.index_add_(0, dst, msg)
            return F.relu(out)

    class RGCNEncoder(nn.Module):
        def __init__(self, num_nodes: int, num_relations: int, dim: int = EMBED_DIM,
                     num_layers: int = NUM_LAYERS):
            super().__init__()
            self.embed = nn.Embedding(num_nodes, dim)
            nn.init.xavier_uniform_(self.embed.weight)
            self.layers = nn.ModuleList(
                [RelGraphConvLayer(dim, dim, num_relations) for _ in range(num_layers)]
            )
            self.rel_embed = nn.Embedding(num_relations, dim)
            nn.init.xavier_uniform_(self.rel_embed.weight)

        def encode(self, edges_by_rel):
            x = self.embed.weight
            for layer in self.layers:
                x = layer(x, edges_by_rel)
            return x

        def score(self, h: "torch.Tensor", rel_ids: "torch.Tensor", t: "torch.Tensor"):
            r = self.rel_embed(rel_ids)
            return (h * r * t).sum(dim=-1)


def _build_edge_data(graph: nx.MultiDiGraph, node_index: Dict[str, int]):
    """Returns (edges_by_rel: {rel_id: (src_tensor, dst_tensor)}, triples: list[(s,r,t)])."""
    edges_by_rel: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(len(RELATION_NAMES))}
    triples = []
    for u, v, data in graph.edges(data=True):
        rel_name = bucket_predicate(data.get("predicate", ""))
        rel_id = RELATION_NAMES.index(rel_name)
        s, t = node_index[u], node_index[v]
        edges_by_rel[rel_id].append((s, t))
        triples.append((s, rel_id, t))
    return edges_by_rel, triples


def train_and_save(graph: nx.MultiDiGraph) -> Optional[dict]:
    """Trains R-GCN node embeddings on the current graph and persists them to
    disk. Returns a small status dict, or None if training was skipped
    (torch unavailable or graph too small to bother)."""
    if not TORCH_AVAILABLE:
        return None
    nodes = list(graph.nodes())
    if len(nodes) < 4 or graph.number_of_edges() < 3:
        return None  # not enough structure for link prediction to learn anything

    node_index = {n: i for i, n in enumerate(nodes)}
    edges_by_rel_raw, triples = _build_edge_data(graph, node_index)
    if not triples:
        return None

    edges_by_rel = {
        rel_id: (
            torch.tensor([s for s, t in pairs], dtype=torch.long) if pairs else torch.empty(0, dtype=torch.long),
            torch.tensor([t for s, t in pairs], dtype=torch.long) if pairs else torch.empty(0, dtype=torch.long),
        )
        for rel_id, pairs in edges_by_rel_raw.items()
    }

    model = RGCNEncoder(len(nodes), len(RELATION_NAMES))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    pos_s = torch.tensor([s for s, r, t in triples], dtype=torch.long)
    pos_r = torch.tensor([r for s, r, t in triples], dtype=torch.long)
    pos_t = torch.tensor([t for s, r, t in triples], dtype=torch.long)
    n_nodes = len(nodes)

    model.train()
    for _epoch in range(EPOCHS):
        optimizer.zero_grad()
        x = model.encode(edges_by_rel)
        pos_score = model.score(x[pos_s], pos_r, x[pos_t])

        # corrupt the tail with a random node for a simple negative sample
        neg_t = torch.randint(0, n_nodes, pos_t.shape)
        neg_score = model.score(x[pos_s], pos_r, x[neg_t])

        labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
        scores = torch.cat([pos_score, neg_score])
        loss = F.binary_cross_entropy_with_logits(scores, labels)

        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_embeddings = model.encode(edges_by_rel).numpy()

    config.GRAPH_STATE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, final_embeddings)
    with open(NODES_PATH, "w") as f:
        json.dump(nodes, f)

    return {"num_nodes": len(nodes), "num_edges": graph.number_of_edges(),
            "final_loss": float(loss.item())}


def load_embeddings() -> Optional[Tuple[List[str], np.ndarray]]:
    if not EMBEDDINGS_PATH.exists() or not NODES_PATH.exists():
        return None
    try:
        embeddings = np.load(EMBEDDINGS_PATH)
        with open(NODES_PATH) as f:
            nodes = json.load(f)
        if len(nodes) != embeddings.shape[0]:
            return None
        return nodes, embeddings
    except Exception:
        return None


def nearest_nodes(seed_names: List[str], k: int = 6, exclude: Optional[set] = None) -> List[str]:
    """Given seed node names already in the graph, returns up to k node names
    whose R-GCN embeddings are closest to the mean of the seeds' embeddings
    (excluding the seeds themselves and anything in `exclude`). Returns []
    if embeddings aren't available."""
    loaded = load_embeddings()
    if loaded is None:
        return []
    nodes, embeddings = loaded
    index = {n: i for i, n in enumerate(nodes)}
    seed_idxs = [index[n] for n in seed_names if n in index]
    if not seed_idxs:
        return []

    exclude = (exclude or set()) | set(seed_names)
    seed_vec = embeddings[seed_idxs].mean(axis=0, keepdims=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    seed_norm = np.linalg.norm(seed_vec, axis=1, keepdims=True) + 1e-8
    sims = (embeddings @ seed_vec.T / (norms * seed_norm)).flatten()

    order = np.argsort(-sims)
    results = []
    for i in order:
        name = nodes[i]
        if name in exclude:
            continue
        results.append(name)
        if len(results) >= k:
            break
    return results