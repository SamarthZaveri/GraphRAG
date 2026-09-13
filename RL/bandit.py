"""
Thin re-export shim: the LinUCBBandit implementation now lives at
backend/app/query_bandit.py, since corpus_router.py needs to LOAD a
trained policy at live query-serving time, not just during offline RL
training -- and backend/app must be able to run standalone without
depending on the RL/ experimentation folder.

train_bandit.py and train_query_bandit.py both still do
`from bandit import LinUCBBandit` unchanged; this file makes that import
resolve to the single real implementation in backend/app instead of a
second copy living here.
"""
from __future__ import annotations
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from backend.app.query_bandit import LinUCBBandit  # noqa: E402,F401