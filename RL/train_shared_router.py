"""Train adaptive routing on numeric and text rewards from shared filings."""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app import config
from app.engines import ENGINE_NAMES
from app.query_bandit import LinUCBBandit
from app.router_features import QUERY_CONTEXT_DIM

ARMS = list(ENGINE_NAMES)
CATEGORIES = ("local", "global", "multi_hop", "conflict")


def load_results(path):
    """Load complete, compatible train rewards without touching global paths."""
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("No training rows")
    signatures = set()
    for row in rows:
        meta = row.get("metadata", {})
        if meta.get("retrieval_version") != config.RETRIEVAL_VERSION:
            raise ValueError("Historical/incompatible retrieval rewards; rerun the train split")
        context = row.get("context", [])
        if (not isinstance(context, list) or len(context) != QUERY_CONTEXT_DIM
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in context)):
            raise ValueError("Invalid context dimension or values")
        for arm in ARMS:
            reward = row.get("reward_" + arm)
            if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not np.isfinite(reward) or not 0 <= reward <= 1:
                raise ValueError(f"Missing/invalid reward for {arm}")
        if meta.get("split") != "train":
            raise ValueError("Router training accepts only train-split rewards")
        signatures.add(tuple(meta.get(k) for k in ("answer_model", "judge_model", "extraction_model", "llm_backend", "dataset", "dataset_fingerprint", "split", "metric")))
    summary_path = path.parent / "summary.json"
    if not summary_path.exists() or json.loads(summary_path.read_text()).get("complete") is not True:
        raise ValueError("Training reward collection is incomplete; rerun the train split")
    if len(signatures) != 1:
        raise ValueError("Mixed model configurations: train from one experiment output directory")
    return rows


def train(rows, alpha=1., seed=0):
    """Simulate online LinUCB learning using only each selected arm's reward."""
    policy = LinUCBBandit(QUERY_CONTEXT_DIM, ARMS, alpha)
    for index in np.random.default_rng(seed).permutation(len(rows)):
        row = rows[index]
        arm, _ = policy.select_arm(row["context"])
        policy.update(arm, row["context"], row["reward_" + arm], weight=row["sample_weight"])
    return policy


def load_weighted(path, numeric_weight=.5):
    if not 0 < numeric_weight < 1:
        raise ValueError("Numeric weight must be strictly between zero and one")
    rows = load_results(path)
    if any(r["metadata"].get("dataset") != "shared_finqa" for r in rows):
        raise ValueError("Use shared-source training rewards")
    if any(r.get("task") not in ("numeric", "text") or (r.get("task") == "text" and r.get("benchmark_category") not in CATEGORIES) for r in rows):
        raise ValueError("Invalid numeric/text task or text category")
    counts = Counter("numeric" if r.get("task") == "numeric" else r.get("benchmark_category") for r in rows)
    if any(not counts[b] for b in ("numeric", *CATEGORIES)):
        raise ValueError("Training requires numeric questions and all four text categories")
    for row in rows:
        bucket = "numeric" if row["task"] == "numeric" else row["benchmark_category"]
        mass = numeric_weight if bucket == "numeric" else (1 - numeric_weight) / 4
        row["sample_weight"] = len(rows) * mass / counts[bucket]
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--numeric-weight", type=float, default=.5)
    parser.add_argument("--alpha", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rows = load_weighted(args.results, args.numeric_weight)
    policy = train(rows, args.alpha, args.seed)
    policy.metadata = {**rows[0]["metadata"], "numeric_weight": args.numeric_weight,
                       "text_category_weight": (1 - args.numeric_weight) / 4}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    policy.save(args.output)
    args.output.with_suffix(".training.json").write_text(json.dumps({"rows": len(rows), "metadata": policy.metadata}, indent=2))
    print("Saved deployable router:", args.output)


if __name__ == "__main__":
    main()
