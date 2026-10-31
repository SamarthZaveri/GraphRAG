"""Train the adaptive router (LinUCB) on train-split rewards from one or more datasets.

Each dataset gets equal weight, and within a dataset the numeric task (if any) gets
--numeric-weight while the four text categories share the rest equally, so no dataset or
category dominates just by having more questions.

--feedback full (the default) updates every arm with its known reward for each question: offline,
all three engines were run on every train question, so there is nothing to explore and discarding
the counterfactual rewards only adds noise. --feedback bandit replays the questions as an online
LinUCB run that sees only the chosen arm's reward; its policy depends on the question order (the
seed and the order of --results), and on dev it ranged from 0.69 to 0.80 across orders, while full
feedback is order-independent.

--latency-weight turns on cost-aware routing: the reward becomes
quality - weight * min(latency / 60 s, 1), so between engines of similar quality the router
learns to prefer the faster one."""
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
MODEL_KEYS = ("retrieval_version", "answer_model", "judge_model", "extraction_model", "query_model", "llm_backend")
LATENCY_SCALE_S = 60.0


def load_results(path):
    """Load complete, compatible train rewards from one dataset's query_results.jsonl."""
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
        signatures.add(tuple(meta.get(k) for k in MODEL_KEYS + ("dataset", "dataset_fingerprint", "metric")))
    summary_path = path.parent / "summary.json"
    if not summary_path.exists() or json.loads(summary_path.read_text()).get("complete") is not True:
        raise ValueError("Training reward collection is incomplete; rerun the train split")
    if len(signatures) != 1:
        raise ValueError("Mixed model configurations: train from one experiment output directory")
    return rows


def train(rows, alpha=1., seed=0, latency_weight=0., feedback="full"):
    """Fit the per-arm LinUCB estimates: from every arm's reward (full) or by simulating online
    learning that observes only each selected arm's reward (bandit)."""
    if feedback not in ("full", "bandit"):
        raise ValueError("Feedback must be 'full' or 'bandit'")
    policy = LinUCBBandit(QUERY_CONTEXT_DIM, ARMS, alpha)
    for index in np.random.default_rng(seed).permutation(len(rows)):
        row = rows[index]
        arms = ARMS if feedback == "full" else [policy.select_arm(row["context"])[0]]
        for arm in arms:
            policy.update(arm, row["context"], effective_reward(row, arm, latency_weight), weight=row["sample_weight"])
    return policy


def effective_reward(row, arm, latency_weight=0.):
    reward = row["reward_" + arm]
    if latency_weight:
        latency = ((row.get("telemetry") or {}).get(arm) or {}).get("latency_ms")
        if latency is None:
            raise ValueError("Cost-aware training needs per-engine telemetry; rerun the benchmark with this version")
        reward -= latency_weight * min(latency / 1000 / LATENCY_SCALE_S, 1.)
    return reward


def load_weighted(paths, numeric_weight=.5):
    if not 0 < numeric_weight < 1:
        raise ValueError("Numeric weight must be strictly between zero and one")
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    groups = [load_results(p) for p in paths]
    models = {tuple(g[0]["metadata"].get(k) for k in MODEL_KEYS) for g in groups}
    if len(models) != 1:
        raise ValueError("Datasets were evaluated with different models or retrieval; use one experiment run")
    if len({g[0]["metadata"]["dataset"] for g in groups}) != len(groups):
        raise ValueError("Each dataset may be given once")
    rows = [r for g in groups for r in g]
    for group in groups:
        if any(r.get("task") not in ("numeric", "text") or (r.get("task") == "text" and r.get("benchmark_category") not in CATEGORIES) for r in group):
            raise ValueError("Invalid numeric/text task or text category")
        counts = Counter("numeric" if r["task"] == "numeric" else r["benchmark_category"] for r in group)
        if any(not counts[c] for c in CATEGORIES):
            raise ValueError(f"{group[0]['metadata']['dataset']}: training requires all four text categories")
        dataset_mass = 1 / len(groups)
        text_mass = (1 - numeric_weight) if counts["numeric"] else 1.
        for row in group:
            bucket = "numeric" if row["task"] == "numeric" else row["benchmark_category"]
            mass = numeric_weight if bucket == "numeric" else text_mass / len(CATEGORIES)
            row["sample_weight"] = len(rows) * dataset_mass * mass / counts[bucket]
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, nargs="+", required=True, help="One query_results.jsonl per dataset")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--numeric-weight", type=float, default=.5)
    parser.add_argument("--latency-weight", type=float, default=0.)
    parser.add_argument("--alpha", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--feedback", choices=("full", "bandit"), default="full")
    args = parser.parse_args()
    import tracking
    rows = load_weighted(args.results, args.numeric_weight)
    policy = train(rows, args.alpha, args.seed, args.latency_weight, args.feedback)
    datasets = {}
    for row in rows:
        datasets[row["metadata"]["dataset"]] = row["metadata"]["dataset_fingerprint"]
    first = rows[0]["metadata"]
    policy.metadata = {**{k: first.get(k) for k in MODEL_KEYS + ("question_model",)}, "split": "train",
                       "datasets": datasets, "dataset": "+".join(sorted(datasets)),
                       "metrics": sorted({r["metadata"]["metric"] for r in rows}),
                       "numeric_weight": args.numeric_weight, "latency_weight": args.latency_weight,
                       "feedback": args.feedback}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    policy.save(args.output)
    settings = {"alpha": args.alpha, "seed": args.seed, "feedback": args.feedback, "numeric_weight": args.numeric_weight, "latency_weight": args.latency_weight}
    args.output.with_suffix(".training.json").write_text(json.dumps({"rows": len(rows), **settings, "metadata": policy.metadata}, indent=2))
    tracking.log_router_training(args.output.parent, args.output, policy.metadata, settings, len(rows))
    version = tracking.register_policy(args.output, {**policy.metadata, "feedback": args.feedback})
    if version:
        print(f"Registered as MLflow model {tracking.REGISTERED_MODEL} version {version}")
    print("Saved deployable router:", args.output, "| datasets:", ", ".join(sorted(datasets)), "| rows:", len(rows))


if __name__ == "__main__":
    main()
