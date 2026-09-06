"""
Trains the LinUCB bandit on real experiment results by simulating the
online process honestly: corpora arrive in a shuffled (random) order, the
policy picks an arm using ONLY what it's learned so far, observes the
reward for the arm it chose (not both -- true bandit feedback, even though
we happen to have recorded both), and updates.

We also know the OTHER arm's reward for every corpus (since experiments
ran both engines), which lets us compute genuine regret for reporting --
that's an evaluation convenience, not something the policy itself gets to
see during training.

Usage:
    python train_bandit.py
    python train_bandit.py --seeds 20   # average over more shuffles for a
                                          # more stable regret estimate at
                                          # this small sample size
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bandit import LinUCBBandit  # noqa: E402
from features import CONTEXT_DIM  # noqa: E402

RESULTS_PATH = Path(__file__).parent / "data" / "experiment_results.jsonl"
POLICY_PATH = Path(__file__).parent / "data" / "trained_bandit.json"
ARMS = ["graphrag", "vector_rag"]


def load_results() -> list[dict]:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"{RESULTS_PATH} not found -- run run_experiments.py first")
    rows = [json.loads(line) for line in RESULTS_PATH.read_text().splitlines() if line.strip()]
    if len(rows) < 4:
        print(f"WARNING: only {len(rows)} corpora recorded. This is genuinely too few to trust "
              f"the trained policy -- treat results as a smoke test, not a validated router.")
    return rows


def run_training(rows: list[dict], alpha: float, seed: int):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    bandit = LinUCBBandit(context_dim=CONTEXT_DIM, arms=ARMS, alpha=alpha)

    log = []
    for idx in order:
        row = rows[idx]
        x = np.array(row["features"])
        true_reward = {"graphrag": row["reward_graphrag"], "vector_rag": row["reward_vector_rag"]}
        best_arm = max(true_reward, key=true_reward.get)

        chosen_arm, scores = bandit.select_arm(x)
        observed_reward = true_reward[chosen_arm]
        regret = true_reward[best_arm] - observed_reward
        bandit.update(chosen_arm, x, observed_reward)

        log.append({
            "corpus": row["corpus"], "chosen": chosen_arm, "best": best_arm,
            "correct": chosen_arm == best_arm, "regret": regret,
        })
    return bandit, log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=1.0, help="LinUCB exploration parameter")
    parser.add_argument("--seeds", type=int, default=10, help="number of random shuffles to average over")
    args = parser.parse_args()

    rows = load_results()
    print(f"Training on {len(rows)} recorded corpora, averaging over {args.seeds} shuffles...\n")

    all_logs = []
    final_bandit = None
    for seed in range(args.seeds):
        bandit, log = run_training(rows, args.alpha, seed)
        all_logs.append(log)
        final_bandit = bandit  # keep the last shuffle's policy as the saved artifact

    accuracy_per_seed = [np.mean([e["correct"] for e in log]) for log in all_logs]
    regret_per_seed = [np.mean([e["regret"] for e in log]) for log in all_logs]

    print(f"Mean arm-selection accuracy (chose the actually-better engine): "
          f"{np.mean(accuracy_per_seed):.1%} (+/- {np.std(accuracy_per_seed):.1%} across shuffles)")
    print(f"Mean regret per decision: {np.mean(regret_per_seed):.3f} (+/- {np.std(regret_per_seed):.3f})")

    # Show what the FINAL trained policy would pick for each real corpus,
    # using its fully-updated beliefs (not the online/exploring choices above)
    print("\nFinal policy's recommendation per corpus (using all data, not just what it saw during that shuffle):")
    for row in rows:
        x = np.array(row["features"])
        arm, scores = final_bandit.select_arm(x)
        actual_best = "graphrag" if row["reward_graphrag"] >= row["reward_vector_rag"] else "vector_rag"
        match = "match" if arm == actual_best else "MISMATCH"
        print(f"  {row['corpus']:35s} recommends={arm:11s} actual_best={actual_best:11s} [{match}]")

    final_bandit.save(POLICY_PATH)
    print(f"\nTrained policy saved to {POLICY_PATH}")

    if len(rows) < 10:
        print(f"\nNOTE: trained on {len(rows)} corpora. Read the accuracy/regret numbers above as "
              f"directional, not a validated production policy -- that needs a larger, held-out set.")


if __name__ == "__main__":
    main()
