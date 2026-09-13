"""
Thin re-export shim: the actual feature computation now lives in
backend/app/router_features.py, since corpus_router.py needs the SAME
doc-level features and query-context builder at live serving time, not
just during offline RL experiments. Keeping two copies risked exactly the
kind of drift that's already bitten this project once (features quietly
disagreeing between where they're computed and where they're used) -- so
this file now just re-exports from the single source of truth in
backend/app, matching the dependency direction every other RL script
already uses (RL depends on backend/app, never the reverse).

run_experiments.py needs ZERO changes because of this move -- it still
does `from features import extract_features, FEATURE_NAMES,
build_query_context` and gets the exact same functions and behavior,
just sourced from one place instead of two.
"""
from __future__ import annotations
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from backend.app.router_features import (  # noqa: E402,F401
    CONTEXT_DIM,
    FEATURE_NAMES,
    extract_features,
    QUERY_CATEGORIES,
    QUERY_CONTEXT_DIM,
    QUERY_FEATURE_NAMES,
    build_query_context,
)