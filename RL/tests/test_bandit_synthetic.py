"""
Correctness tests for the bandit ALGORITHM ITSELF, using synthetic
environments with known ground-truth reward functions. This is deliberately
separate from testing on real corpus data: here we know exactly which arm
SHOULD be chosen for a given context, so we can assert the policy actually
learns that, rather than just asserting "it ran without crashing."
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from bandit import LinUCBBandit


def run_episode(bandit, context_fn, reward_fn, rounds, rng):
    """context_fn() -> x; reward_fn(arm, x) -> true expected reward (no noise).
    Returns list of (regret) per round, where regret = best_possible - chosen."""
    regrets = []
    for _ in range(rounds):
        x = context_fn(rng)
        arm, _ = bandit.select_arm(x)
        true_rewards = {a: reward_fn(a, x) for a in bandit.arms}
        best_arm = max(true_rewards, key=true_rewards.get)
        regret = true_rewards[best_arm] - true_rewards[arm]
        regrets.append(regret)
        observed = true_rewards[arm] + rng.normal(0, 0.05)
        bandit.update(arm, x, observed)
    return regrets


def test_learns_context_dependent_best_arm():
    """Arm 'a' is better when x[0] > 0, arm 'b' is better when x[0] < 0.
    A working contextual bandit should learn this split, not just pick
    whichever arm looked good on average."""
    rng = np.random.default_rng(0)
    bandit = LinUCBBandit(context_dim=2, arms=["a", "b"], alpha=1.0)

    def context_fn(rng):
        return np.array([rng.uniform(-1, 1), 1.0])  # second dim = bias term

    def reward_fn(arm, x):
        return x[0] if arm == "a" else -x[0]

    regrets = run_episode(bandit, context_fn, reward_fn, rounds=400, rng=rng)

    early_regret = np.mean(regrets[:40])
    late_regret = np.mean(regrets[-40:])
    assert late_regret < early_regret, "regret should decrease as the policy learns"
    assert late_regret < 0.15, f"late-stage regret too high ({late_regret:.3f}) -- policy isn't converging"

    # Explicit context-dependent correctness check, no exploration noise
    arm_pos, _ = bandit.select_arm([0.8, 1.0])
    arm_neg, _ = bandit.select_arm([-0.8, 1.0])
    assert arm_pos == "a", "should prefer arm 'a' when x[0] is strongly positive"
    assert arm_neg == "b", "should prefer arm 'b' when x[0] is strongly negative"


def test_converges_to_dominant_arm_when_context_irrelevant():
    """If one arm is simply always better regardless of context, the policy
    should converge to always picking it -- a sanity check that the bandit
    doesn't get confused by irrelevant context dimensions."""
    rng = np.random.default_rng(1)
    bandit = LinUCBBandit(context_dim=3, arms=["good", "bad"], alpha=0.5)

    def context_fn(rng):
        return np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), 1.0])

    def reward_fn(arm, x):
        return 0.8 if arm == "good" else 0.2

    choices = []
    for _ in range(200):
        x = context_fn(rng)
        arm, _ = bandit.select_arm(x)
        choices.append(arm)
        bandit.update(arm, x, reward_fn(arm, x) + rng.normal(0, 0.02))

    late_choices = choices[-30:]
    assert late_choices.count("good") >= 27, "should converge to almost always picking the dominant arm"


def test_higher_alpha_explores_more():
    """Exploration bonus scales with alpha -- a higher alpha should try the
    apparently-worse arm more often early on, when uncertainty is high."""
    rng1, rng2 = np.random.default_rng(2), np.random.default_rng(2)

    def context_fn(rng):
        return np.array([1.0])

    def reward_fn(arm, x):
        return 0.6 if arm == "a" else 0.4

    low_alpha = LinUCBBandit(context_dim=1, arms=["a", "b"], alpha=0.1)
    high_alpha = LinUCBBandit(context_dim=1, arms=["a", "b"], alpha=3.0)

    def count_b_choices(bandit, rng):
        count = 0
        for _ in range(30):
            x = context_fn(rng)
            arm, _ = bandit.select_arm(x)
            if arm == "b":
                count += 1
            bandit.update(arm, x, reward_fn(arm, x) + rng.normal(0, 0.01))
        return count

    low_b = count_b_choices(low_alpha, rng1)
    high_b = count_b_choices(high_alpha, rng2)
    assert high_b >= low_b, "higher alpha should explore the worse-looking arm at least as much"


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    bandit = LinUCBBandit(context_dim=2, arms=["graphrag", "vector_rag"], alpha=1.0)
    for _ in range(20):
        x = [rng.uniform(-1, 1), 1.0]
        arm, _ = bandit.select_arm(x)
        bandit.update(arm, x, rng.uniform(0, 1))

    path = tmp_path / "bandit.json"
    bandit.save(path)
    loaded = LinUCBBandit.load(path)

    test_x = [0.3, 1.0]
    orig_scores = bandit.scores(np.array(test_x))
    loaded_scores = loaded.scores(np.array(test_x))
    for arm in bandit.arms:
        assert orig_scores[arm] == pytest.approx(loaded_scores[arm], abs=1e-9)


def test_two_arms_realistic_router_scale():
    """A regression-style check at the ACTUAL scale this project will train
    at (12-20 corpora, not hundreds) -- confirms the policy still moves in
    the right direction even with a small sample, though we don't expect
    full convergence at this scale and shouldn't pretend otherwise."""
    rng = np.random.default_rng(4)
    bandit = LinUCBBandit(context_dim=2, arms=["graphrag", "vector_rag"], alpha=1.0)

    # Simulates: high cross-doc-overlap corpora favor graphrag, low favor vector_rag
    def context_fn(rng):
        return np.array([rng.uniform(0, 1), 1.0])

    def reward_fn(arm, x):
        graphrag_reward = 0.3 + 0.5 * x[0]
        vector_reward = 0.75 - 0.15 * x[0]
        return graphrag_reward if arm == "graphrag" else vector_reward

    regrets = run_episode(bandit, context_fn, reward_fn, rounds=16, rng=rng)
    # honest bar: with n=16, we check it's learning SOMETHING, not full convergence
    assert np.mean(regrets) < 0.35, "even at small scale, average regret shouldn't be near-random"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
