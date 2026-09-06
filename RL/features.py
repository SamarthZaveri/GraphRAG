"""
Extracts the bandit's context feature vector from an ingested corpus,
reusing the exact same signals the rule-based router in the main app
computes (backend/app/corpus_router.py) -- this keeps the bandit and the
rule-based fallback speaking the same language, and means the bandit is
learning to weight/combine signals a human already picked as relevant,
rather than starting from nothing.

Feature vector: [cross_doc_entity_fraction, community_modularity,
rgcn_val_auc, num_docs_normalized, bias]
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

BACKEND_ROOT = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

CONTEXT_DIM = 5
FEATURE_NAMES = ["cross_doc_entity_fraction", "community_modularity", "rgcn_val_auc", "num_docs_normalized", "bias"]


def extract_features(store, modularity: float | None, rgcn_val_auc: float | None) -> np.ndarray:
    total_entities = max(store.graph.number_of_nodes(), 1)
    shared = sum(1 for _, d in store.graph.nodes(data=True) if len(d.get("source_docs", [])) >= 2)
    cross_doc_fraction = min(shared / total_entities, 1.0)

    mod = 0.0 if modularity is None else max(0.0, min(modularity, 1.0))
    auc = 0.5 if rgcn_val_auc is None else max(0.0, min(rgcn_val_auc, 1.0))

    num_docs = len(store.ingested_doc_ids())
    num_docs_norm = min(num_docs / 10.0, 1.0)

    return np.array([cross_doc_fraction, mod, auc, num_docs_norm, 1.0], dtype=float)
