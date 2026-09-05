"""
Self-supervised R-GCN node embeddings.

Two upgrades over a naive implementation:

1. SEMANTIC INITIALIZATION. A free-parameter nn.Embedding table learns
   node vectors purely from graph structure -- weak signal on a small graph
   with limited edges. Instead, each node's starting feature is its local
   text embedding (name + type + description, via text_embeddings.py), then
   a learnable projection + the relational graph-conv layers refine that
   semantic starting point using graph structure on top. This is standard
   practice for text-attributed graphs: don't throw away the text signal
   just because you also have a graph.

2. VALIDATION. A held-out slice of real edges is scored against random
   negatives after training (link-prediction AUC), computed with a plain
   rank-based formula (no new dependency). This is a genuine evaluation of
   whether the embeddings learned anything, not just "loss went down."
   The AUC is persisted and used downstream by the corpus-health router.

Trained via link prediction since there's no labeled data: node vectors are
learned such that real edges score higher than randomly-corrupted negative
pairs under a DistMult decoder, refined through relation-bucketed graph
convolution layers (raw predicates are free text, so they're bucketed into
a small fixed relation-type set first).

The resulting embeddings are transductive (only defined for nodes already
in the graph) and used for graph-internal retrieval tasks -- local-search
neighborhood expansion in query_engine.py. Linking an arbitrary NEW query
string to a node is a different problem, handled by the general-purpose
text encoder in text_embeddings.py instead.

Entirely optional: if PyTorch isn't installed, every public function here
returns None / a no-op, and the rest of the app works exactly as before.
"""
from __future__ import annotations
import json
import random
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
METRICS_PATH = config.GRAPH_STATE_DIR / "rgcn_metrics.json"

EMBED_DIM = 48
NUM_LAYERS = 2
EPOCHS = 120
LR = 0.02
DROPOUT = 0.1
NEG_PER_POS = 2          # negative samples drawn per positive triple
VAL_FRACTION = 0.15      # held-out edges for link-prediction AUC
MIN_EDGES_FOR_VAL = 20   # below this, skip validation (too noisy to mean anything)

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
        def __init__(self, in_dim: int, out_dim: int, num_relations: int, dropout: float = DROPOUT):
            super().__init__()
            self.rel_weights = nn.ModuleList(
                [nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_relations)]
            )
            self.self_weight = nn.Linear(in_dim, out_dim, bias=False)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: "torch.Tensor", edges_by_rel: Dict[int, Tuple["torch.Tensor", "torch.Tensor"]]):
            out = self.self_weight(x)
            for rel_id, (src, dst) in edges_by_rel.items():
                if src.numel() == 0:
                    continue
                msg = self.rel_weights[rel_id](x[src])
                out.index_add_(0, dst, msg)
            return self.dropout(F.relu(out))

    class RGCNEncoder(nn.Module):
        """Semantic-init encoder: starts from fixed text-embedding features,
        projects into the working dimension, then refines via relational
        graph convolution. Falls back to a learned free embedding table if
        semantic features aren't available (e.g. embedding backend offline)."""

        def __init__(self, num_relations: int, semantic_features: Optional["torch.Tensor"],
                     num_nodes: int, dim: int = EMBED_DIM, num_layers: int = NUM_LAYERS):
            super().__init__()
            if semantic_features is not None:
                self.register_buffer("semantic_features", semantic_features)
                self.input_proj = nn.Linear(semantic_features.shape[1], dim)
                self.free_embed = None
            else:
                self.semantic_features = None
                self.input_proj = None
                self.free_embed = nn.Embedding(num_nodes, dim)
                nn.init.xavier_uniform_(self.free_embed.weight)
            self.layers = nn.ModuleList(
                [RelGraphConvLayer(dim, dim, num_relations) for _ in range(num_layers)]
            )
            self.rel_embed = nn.Embedding(num_relations, dim)
            nn.init.xavier_uniform_(self.rel_embed.weight)

        def encode(self, edges_by_rel):
            x = self.input_proj(self.semantic_features) if self.semantic_features is not None \
                else self.free_embed.weight
            for layer in self.layers:
                x = layer(x, edges_by_rel)
            return x

        def score(self, h: "torch.Tensor", rel_ids: "torch.Tensor", t: "torch.Tensor"):
            r = self.rel_embed(rel_ids)
            return (h * r * t).sum(dim=-1)


def _build_edge_data(graph: nx.MultiDiGraph, node_index: Dict[str, int]):
    edges_by_rel: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(len(RELATION_NAMES))}
    triples = []
    for u, v, data in graph.edges(data=True):
        rel_name = bucket_predicate(data.get("predicate", ""))
        rel_id = RELATION_NAMES.index(rel_name)
        s, t = node_index[u], node_index[v]
        edges_by_rel[rel_id].append((s, t))
        triples.append((s, rel_id, t))
    return edges_by_rel, triples


def _get_semantic_features(nodes: List[str], graph: nx.MultiDiGraph) -> Optional["np.ndarray"]:
    try:
        from .text_embeddings import embed_texts
        texts = [f"{n} ({graph.nodes[n].get('type', 'Other')}): {graph.nodes[n].get('description', '')}"
                 for n in nodes]
        vecs = embed_texts(texts)
        if vecs.shape[0] != len(nodes) or vecs.shape[1] == 0:
            return None
        return vecs
    except Exception:
        return None


def _rank_auc(pos_scores: "np.ndarray", neg_scores: "np.ndarray") -> float:
    """Rank-based AUC, no sklearn/scipy dependency."""
    all_scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    order = np.argsort(all_scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(all_scores) + 1)
    n_pos, n_neg = labels.sum(), len(labels) - labels.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    pos_rank_sum = ranks[labels == 1].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def train_and_save(graph: nx.MultiDiGraph) -> Optional[dict]:
    """Trains R-GCN node embeddings on the current graph and persists them
    (plus validation metrics) to disk. Returns a status dict, or None if
    training was skipped (torch unavailable or graph too small)."""
    if not TORCH_AVAILABLE:
        return None
    nodes = list(graph.nodes())
    if len(nodes) < 4 or graph.number_of_edges() < 3:
        return None

    node_index = {n: i for i, n in enumerate(nodes)}
    edges_by_rel_raw, all_triples = _build_edge_data(graph, node_index)
    if not all_triples:
        return None

    # Held-out validation split for link-prediction AUC
    rng = random.Random(42)
    shuffled = all_triples[:]
    rng.shuffle(shuffled)
    do_val = len(shuffled) >= MIN_EDGES_FOR_VAL
    n_val = max(1, int(len(shuffled) * VAL_FRACTION)) if do_val else 0
    val_triples = shuffled[:n_val]
    train_triples = shuffled[n_val:] if do_val else shuffled

    # message passing uses the FULL graph structure (transductive setting);
    # only the loss/eval triples are split, per standard KG-embedding practice
    edges_by_rel = {
        rel_id: (
            torch.tensor([s for s, t in pairs], dtype=torch.long) if pairs else torch.empty(0, dtype=torch.long),
            torch.tensor([t for s, t in pairs], dtype=torch.long) if pairs else torch.empty(0, dtype=torch.long),
        )
        for rel_id, pairs in edges_by_rel_raw.items()
    }

    semantic_np = _get_semantic_features(nodes, graph)
    semantic_features = torch.tensor(semantic_np, dtype=torch.float32) if semantic_np is not None else None

    model = RGCNEncoder(len(RELATION_NAMES), semantic_features, len(nodes))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    n_nodes = len(nodes)
    train_s = torch.tensor([s for s, r, t in train_triples], dtype=torch.long)
    train_r = torch.tensor([r for s, r, t in train_triples], dtype=torch.long)
    train_t = torch.tensor([t for s, r, t in train_triples], dtype=torch.long)

    model.train()
    final_loss = None
    for _epoch in range(EPOCHS):
        optimizer.zero_grad()
        x = model.encode(edges_by_rel)
        pos_score = model.score(x[train_s], train_r, x[train_t])

        neg_scores = []
        for _ in range(NEG_PER_POS):
            corrupt_head = torch.rand(train_s.shape) < 0.5
            neg_s = torch.where(corrupt_head, torch.randint(0, n_nodes, train_s.shape), train_s)
            neg_t = torch.where(~corrupt_head, torch.randint(0, n_nodes, train_t.shape), train_t)
            neg_scores.append(model.score(x[neg_s], train_r, x[neg_t]))
        neg_score = torch.cat(neg_scores)

        labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
        scores = torch.cat([pos_score, neg_score])
        loss = F.binary_cross_entropy_with_logits(scores, labels)

        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    model.eval()
    val_auc = None
    with torch.no_grad():
        x = model.encode(edges_by_rel)
        if do_val and val_triples:
            val_s = torch.tensor([s for s, r, t in val_triples], dtype=torch.long)
            val_r = torch.tensor([r for s, r, t in val_triples], dtype=torch.long)
            val_t = torch.tensor([t for s, r, t in val_triples], dtype=torch.long)
            val_pos = model.score(x[val_s], val_r, x[val_t]).numpy()
            neg_t = torch.randint(0, n_nodes, val_t.shape)
            val_neg = model.score(x[val_s], val_r, x[neg_t]).numpy()
            val_auc = _rank_auc(val_pos, val_neg)
        final_embeddings = x.numpy()

    config.GRAPH_STATE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, final_embeddings)
    with open(NODES_PATH, "w") as f:
        json.dump(nodes, f)

    metrics = {
        "num_nodes": len(nodes), "num_edges": graph.number_of_edges(),
        "final_loss": float(final_loss), "val_auc": val_auc,
        "used_semantic_init": semantic_features is not None,
        "num_val_edges": len(val_triples),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f)

    return metrics


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


def load_metrics() -> Optional[dict]:
    if not METRICS_PATH.exists():
        return None
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def nearest_nodes(seed_names: List[str], k: int = 6, exclude: Optional[set] = None) -> List[str]:
    """Given seed node names already in the graph, returns up to k node names
    whose R-GCN embeddings are closest to the mean of the seeds' embeddings
    (excluding the seeds themselves and anything in `exclude`). Returns []
    if embeddings aren't available or k <= 0 (ablation off-switch)."""
    if k <= 0:
        return []
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