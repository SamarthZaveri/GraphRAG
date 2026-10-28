"""Statistical analysis of a benchmark split: engine and router rewards with
bootstrap confidence intervals, routing behaviour, retrieval hit rates and
abstention. Reads <run>/<split>/results.json and writes <run>/<split>/analysis.json.

Usage: python RL/analyze_results.py --run-dir RL/data/local-14b --split dev"""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.engines import ENGINE_NAMES
from app.telemetry import ABSTAIN_PATTERN

CATEGORIES = ("local", "global", "multi_hop", "conflict")
RESAMPLES = 10000


def bootstrap_ci(values, rng, resamples=RESAMPLES):
    """Mean with a percentile-bootstrap 95% CI; resampling questions captures small-n uncertainty."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {"mean": None, "ci95": None, "n": 0}
    idx = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[idx].mean(axis=1)
    return {"mean": round(float(values.mean()), 4),
            "ci95": [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)],
            "n": int(len(values))}


def paired_difference(a, b, rng, resamples=RESAMPLES):
    """Paired bootstrap of mean(a - b) over the same questions; p_better = share of resamples > 0."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if len(diff) == 0:
        return None
    idx = rng.integers(0, len(diff), size=(resamples, len(diff)))
    means = diff[idx].mean(axis=1)
    return {"mean_diff": round(float(diff.mean()), 4),
            "ci95": [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)],
            "p_better": round(float((means > 0).mean()), 4),
            "significant": bool(np.percentile(means, 2.5) > 0 or np.percentile(means, 97.5) < 0)}


def gold_docs(manifest, split):
    out = {}
    for bundle in manifest["splits"][split]["bundles"]:
        for q in bundle["questions"]:
            docs = {e["doc_id"] for e in q.get("evidence", []) if e.get("doc_id")}
            if q.get("source_doc_id"):
                docs.add(q["source_doc_id"])
            out[q["id"]] = docs
    return out


def retrieval_metrics(records, engine, gold):
    """Document-level hit@k and MRR from each engine's ranked citations against evidence documents."""
    hits, rr = [], []
    for r in records:
        target = gold.get(r["question_id"])
        if not target:
            continue
        ranked = list(dict.fromkeys(c["doc_id"] for c in r["answers"][engine].get("citations", [])))
        rank = next((i + 1 for i, d in enumerate(ranked) if d in target), None)
        hits.append(rank is not None)
        rr.append(1 / rank if rank else 0.0)
    return {"doc_hit_rate": round(float(np.mean(hits)), 4) if hits else None,
            "doc_mrr": round(float(np.mean(rr)), 4) if rr else None, "n": len(hits)}


def analyze(records, gold, seed=0):
    rng = np.random.default_rng(seed)
    routed = all("router_engine" in r for r in records) and records
    rewards = {e: [r["reward_" + e] for r in records] for e in ENGINE_NAMES}
    oracle = [max(r["reward_" + e] for e in ENGINE_NAMES) for r in records]
    policies = {**{f"always_{e}": rewards[e] for e in ENGINE_NAMES},
                "random": [float(np.mean([r["reward_" + e] for e in ENGINE_NAMES])) for r in records],
                "oracle": oracle}
    if routed:
        policies["router"] = [r["router_reward"] for r in records]
    result = {"n": len(records), "policies": {k: bootstrap_ci(v, rng) for k, v in policies.items()}}

    best_single = max(ENGINE_NAMES, key=lambda e: np.mean(rewards[e]))
    result["best_single_engine"] = best_single
    winners = Counter()
    for r in records:
        scores = {e: r["reward_" + e] for e in ENGINE_NAMES}
        top = max(scores.values())
        best = [e for e, s in scores.items() if s == top]
        winners[best[0] if len(best) == 1 else "tie"] += 1
    result["outright_wins"] = dict(winners)

    if routed:
        choices = Counter(r["router_engine"] for r in records)
        result["router"] = {
            "selection_share": {e: round(choices[e] / len(records), 4) for e in ENGINE_NAMES},
            # Picked an engine tied for the best reward on that question.
            "optimal_pick_rate": round(float(np.mean([r["router_reward"] == o for r, o in zip(records, oracle)])), 4),
            "mean_regret_vs_oracle": round(float(np.mean(oracle) - np.mean(policies["router"])), 4),
            "vs_engines": {e: paired_difference(policies["router"], rewards[e], rng) for e in ENGINE_NAMES},
        }
    else:
        result["engine_vs_best"] = {e: paired_difference(rewards[e], rewards[best_single], rng)
                                    for e in ENGINE_NAMES if e != best_single}

    def breakdown(subset):
        out = {e: bootstrap_ci([r["reward_" + e] for r in subset], rng) for e in ENGINE_NAMES}
        if routed:
            out["router"] = bootstrap_ci([r["router_reward"] for r in subset], rng)
            out["router_selection"] = dict(Counter(r["router_engine"] for r in subset))
        return out
    result["by_category_all_tasks"] = {c: breakdown([r for r in records if r.get("benchmark_category", r["category"]) == c])
                             for c in CATEGORIES}
    result["by_task"] = {t: breakdown([r for r in records if r.get("task") == t]) for t in ("numeric", "text")}
    result["retrieval"] = {e: retrieval_metrics(records, e, gold) for e in ENGINE_NAMES}
    result["abstention_rate"] = {e: round(float(np.mean([bool(ABSTAIN_PATTERN.search(r["answers"][e]["answer"] or ""))
                                                          for r in records])), 4) for e in ENGINE_NAMES}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("RL/data/local-14b"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("RL/data/shared_finqa"))
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    args = parser.parse_args()
    records = json.loads((args.run_dir / args.split / "results.json").read_text())
    manifest = json.loads((args.dataset_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    result = {"split": args.split, "metadata": records[0]["metadata"] if records else None,
              **analyze(records, gold_docs(manifest, args.split))}
    (args.run_dir / args.split / "analysis.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
