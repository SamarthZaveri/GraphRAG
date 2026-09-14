"""
LinUCB contextual bandit (Li et al. 2010, "A Contextual-Bandit Approach to
Personalized News Article Recommendation").

RELOCATION NOTE: this used to live at RL/bandit.py. It's moved here because
corpus_router.py needs to LOAD a trained policy at live query-serving time,
not just during offline RL training -- and backend/app must be able to run
standalone without depending on the RL/ experimentation folder. RL/bandit.py
is now a thin re-export shim pointing here, so train_bandit.py and
train_query_bandit.py both still work unchanged.

Why this algorithm and not deep RL: we have on the order of a dozen-to-a-few-
dozen corpora (or, for the per-question router, tens to ~100 individual
questions) to learn from -- nowhere near enough to train a neural policy
safely. LinUCB maintains an explicit uncertainty estimate per arm (via the
A matrix, effectively a running covariance) and its exploration bonus
shrinks as evidence accumulates -- it's the right tool for exactly this data
scale, and it's a real, standard, citable algorithm, not a toy simplification
invented for this project.

At TRAINING time, select_arm() (mean + UCB exploration bonus) is used, to
simulate honest online learning. At SERVING time (corpus_router.route_query),
predicted_reward() is used instead for each arm and the max is taken --
pure exploitation of the learned theta, no exploration bonus, since a live
request isn't an opportunity to explore, it's a decision that needs the
best current estimate.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


class LinUCBBandit:
    def __init__(self, context_dim: int, arms: List[str], alpha: float = 1.0):
        self.context_dim = context_dim
        self.arms = list(arms)
        self.alpha = alpha
        self.A: Dict[str, np.ndarray] = {a: np.identity(context_dim) for a in self.arms}
        self.b: Dict[str, np.ndarray] = {a: np.zeros(context_dim) for a in self.arms}

    def _theta(self, arm: str) -> np.ndarray:
        return np.linalg.solve(self.A[arm], self.b[arm])

    def scores(self, x: np.ndarray) -> Dict[str, float]:
        x = np.asarray(x, dtype=float)
        out = {}
        for a in self.arms:
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            mean = float(theta @ x)
            bonus = self.alpha * float(np.sqrt(max(x @ A_inv @ x, 0.0)))
            out[a] = mean + bonus
        return out

    def select_arm(self, x) -> Tuple[str, Dict[str, float]]:
        """Returns (chosen_arm, per_arm_scores). Ties broken by arm order."""
        scores = self.scores(np.asarray(x, dtype=float))
        best = max(scores, key=lambda a: (scores[a], -self.arms.index(a)))
        return best, scores

    def predicted_reward(self, arm: str, x) -> float:
        x = np.asarray(x, dtype=float)
        return float(self._theta(arm) @ x)

    def update(self, arm: str, x, reward: float):
        x = np.asarray(x, dtype=float)
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x

    def save(self, path: str | Path):
        data = {
            "context_dim": self.context_dim, "arms": self.arms, "alpha": self.alpha,
            "A": {a: self.A[a].tolist() for a in self.arms},
            "b": {a: self.b[a].tolist() for a in self.arms},
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> "LinUCBBandit":
        data = json.loads(Path(path).read_text())
        bandit = cls(data["context_dim"], data["arms"], data["alpha"])
        for a in bandit.arms:
            bandit.A[a] = np.array(data["A"][a])
            bandit.b[a] = np.array(data["b"][a])
        return bandit