"""LinUCB contextual bandit shared by offline training and app routing.

Training selects arms using the uncertainty bonus and updates the selected
arm from its observed reward. Serving selects the greatest predicted reward
without exploration or updates. Policies persist as JSON."""
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
        self.metadata = {}
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

    def update(self, arm: str, x, reward: float, weight: float = 1.0):
        x = np.asarray(x, dtype=float)
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("Sample weight must be positive and finite")
        self.A[arm] += weight * np.outer(x, x)
        self.b[arm] += weight * reward * x

    def save(self, path: str | Path):
        data = {
            "context_dim": self.context_dim, "arms": self.arms, "alpha": self.alpha,
            "metadata": self.metadata,
            "A": {a: self.A[a].tolist() for a in self.arms},
            "b": {a: self.b[a].tolist() for a in self.arms},
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> "LinUCBBandit":
        data = json.loads(Path(path).read_text())
        bandit = cls(data["context_dim"], data["arms"], data["alpha"])
        bandit.metadata = data.get("metadata", {})
        for a in bandit.arms:
            bandit.A[a] = np.array(data["A"][a])
            bandit.b[a] = np.array(data["b"][a])
        return bandit