"""
Trains the per-QUESTION contextual bandit on query_results.jsonl (one row
per (corpus, question) pair, produced by run_experiments.py) -- NOT the
per-corpus train_bandit.py, which real evaluation data showed converges to
a near-constant policy (corpus-level structure didn't predict which
engine wins on any individual question; see router_features.py's revision
notes for why).

Context here is 8-dim: [doc structure features (3), query category
one-hot (4), bias] -- see router_features.build_query_context(). Same
honest simulated-online-training method as train_bandit.py: rows arrive
in shuffled order, the policy picks using only what it's learned so far,
observes the reward for the arm it picked, and updates. Regret is
computed using both known rewards, for reporting only.

IMPORTANT: this saves the trained policy directly into the BACKEND's
state directory (config.GRAPH_STATE_DIR / "query_bandit.json"), not
RL/data/ -- that's the exact path corpus_router.route_query() reads from
at live serving time, so there's no manual copy step between training and
the app picking up the new policy. Just restart/re-request after training.

Usage:
    python train_query_bandit.py
    python train_query_bandit.py --seeds 20
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from backend.app import config as backend_config  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from bandit import LinUCBBandit  # noqa: E402
from features import QUERY_CONTEXT_DIM  # noqa: E402

RESULTS_PATH = Path(__file__).parent / "data" / "query_results.jsonl"
POLICY_PATH = backend_config.GRAPH_STATE_DIR / "query_bandit.json"
ARMS = ["graphrag", "vector_rag"]


def load_results() -> list[dict]:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"{RESULTS_PATH} not found -- run run_experiments.py first "
            f"(it now writes this file automatically alongside experiment_results.jsonl)"
        )
    rows = [json.loads(line) for line in RESULTS_PATH.read_text().splitlines() if line.strip()]
    if len(rows) < 20:
        print(f"WARNING: only {len(rows)} per-question rows recorded. This is more than the "
              f"old 12-corpus set, but still treat results as directional at this scale, not a "
              f"fully validated production policy.")
    return rows


def run_training(rows: list[dict], alpha: float, seed: int):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    bandit = LinUCBBandit(context_dim=QUERY_CONTEXT_DIM, arms=ARMS, alpha=alpha)

    log = []
    for idx in order:
        row = rows[idx]
        x = np.array(row["context"])
        true_reward = {"graphrag": row["reward_graphrag"], "vector_rag": row["reward_vector_rag"]}
        best_arm = max(true_reward, key=true_reward.get)

        chosen_arm, scores = bandit.select_arm(x)
        observed_reward = true_reward[chosen_arm]
        regret = true_reward[best_arm] - observed_reward
        bandit.update(chosen_arm, x, observed_reward)

        log.append({
            "corpus": row["corpus"], "question": row["question"], "category": row["category"],
            "chosen": chosen_arm, "best": best_arm, "correct": chosen_arm == best_arm,
            "regret": regret,
        })
    return bandit, log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=1.0, help="LinUCB exploration parameter")
    parser.add_argument("--seeds", type=int, default=10, help="number of random shuffles to average over")
    args = parser.parse_args()

    rows = load_results()
    print(f"Training on {len(rows)} recorded (corpus, question) rows, "
          f"averaging over {args.seeds} shuffles...\n")

    all_logs = []
    final_bandit = None
    for seed in range(args.seeds):
        bandit, log = run_training(rows, args.alpha, seed)
        all_logs.append(log)
        final_bandit = bandit

    accuracy_per_seed = [np.mean([e["correct"] for e in log]) for log in all_logs]
    regret_per_seed = [np.mean([e["regret"] for e in log]) for log in all_logs]

    print(f"Mean arm-selection accuracy (chose the actually-better engine): "
          f"{np.mean(accuracy_per_seed):.1%} (+/- {np.std(accuracy_per_seed):.1%} across shuffles)")
    print(f"Mean regret per decision: {np.mean(regret_per_seed):.3f} (+/- {np.std(regret_per_seed):.3f})")

    # Per-category breakdown -- this is the actual point of moving to a
    # per-query router: does accuracy differ by question type, not just
    # overall? A flat ~same accuracy across all 4 categories would suggest
    # category isn't adding real signal either.
    last_log = all_logs[-1]
    by_category = defaultdict(list)
    for e in last_log:
        by_category[e["category"]].append(e["correct"])
    print("\nAccuracy by question category (last shuffle):")
    for cat, corrects in sorted(by_category.items()):
        print(f"  {cat:12s} {np.mean(corrects):.1%}  ({len(corrects)} questions)")

    # Per-category predicted reward at that category's median doc-structure
    # point, using the fully-trained final policy -- cheaper to read than
    # a 70+ row table, and shows whether the policy learned to differentiate
    # by category at all.
    print("\nFinal policy's predicted reward by category (median doc-structure per category):")
    contexts_by_cat = defaultdict(list)
    for row in rows:
        contexts_by_cat[row["category"]].append(row["context"])
    for cat, contexts in sorted(contexts_by_cat.items()):
        median_context = np.median(np.array(contexts), axis=0)
        rewards = {arm: final_bandit.predicted_reward(arm, median_context) for arm in ARMS}
        best = max(rewards, key=rewards.get)
        print(f"  {cat:12s} graphrag={rewards['graphrag']:.3f}  vector_rag={rewards['vector_rag']:.3f}"
              f"  -> {best}")

    POLICY_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_bandit.save(POLICY_PATH)
    print(f"\nTrained query-level policy saved to {POLICY_PATH}")
    print("(this is the exact path corpus_router.route_query() reads at serving time -- "
          "no manual copy step needed, just start/restart the backend)")

    if len(rows) < 40:
        print(f"\nNOTE: trained on {len(rows)} question-level rows. Read accuracy/regret and "
              f"the per-category breakdown as directional signal about whether category helps "
              f"at all, not as a validated production accuracy number.")


if __name__ == "__main__":
    main()