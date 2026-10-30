"""Export a benchmark run as static JSON for the replay site (frontend/data/replay.json).

The replay site is the frontend with no backend: benchmark results, recorded showcase answers from
all three engines with judge scores and telemetry, and the router's decision for each. Re-run after
each split finishes, then redeploy the frontend:

    python scripts/export_replay.py                       # RL/data/runs/mixed -> frontend/data/replay.json
    python scripts/export_replay.py --run-dir RL/data/runs/7b

Showcase questions prefer held-out splits (test, then dev) and questions where the engines disagree.
Only source snippets, generated questions and model outputs are exported; no credentials or paths.
"""
from __future__ import annotations
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

SPLITS = ("train", "dev", "test")
ENGINES = ("graphrag", "vector_rag", "hybrid_rag")
CATEGORIES = ("local", "global", "multi_hop", "conflict")
SNIPPET_CHARS = 320
CITATIONS_PER_ANSWER = 6


def load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def results_payload(run):
    """Same shape as GET /api/rl/results."""
    def pair(base, split):
        return {"summary": load(base / split / "summary.json"), "analysis": load(base / split / "analysis.json")}
    datasets = sorted(p.name for p in run.iterdir() if p.is_dir() and p.name not in SPLITS
                      and any((p / s / "summary.json").exists() for s in SPLITS))
    policy = load(run / "policy.json")
    return {
        "splits": {s: pair(run, s) for s in SPLITS},
        "datasets": {d: {s: pair(run / d, s) for s in SPLITS} for d in datasets},
        "policy": {k: policy.get(k) for k in ("arms", "alpha", "context_dim", "metadata")} if policy else None,
    }, datasets


def predictor(run):
    from app.query_bandit import LinUCBBandit
    try:
        policy = LinUCBBandit.load(run / "policy.json")
    except (OSError, ValueError, KeyError):
        return None
    def predict(context):
        if len(context or []) != policy.context_dim:
            return None
        return {arm: round(policy.predicted_reward(arm, context), 4) for arm in policy.arms}
    return predict


def records_by_split(run, datasets):
    out = {}
    for split in SPLITS:
        rows = []
        for d in datasets:
            for r in load(run / d / split / "results.json") or []:
                rows.append({**r, "dataset": d, "split": split})
        if rows:
            out[split] = rows
    return out


def trim_citations(citations):
    return [{"doc_id": c.get("doc_id"), "chunk_id": c.get("chunk_id"), "snippet": (c.get("snippet") or "")[:SNIPPET_CHARS]}
            for c in (citations or [])[:CITATIONS_PER_ANSWER]]


def showcase_entry(r, predict):
    predicted = predict(r.get("context")) if predict else None
    rewards = {e: r.get("reward_" + e) for e in ENGINES}
    chosen = r.get("router_engine") or (max(predicted, key=predicted.get) if predicted else None)
    answers = {}
    for e in ENGINES:
        a = r["answers"].get(e) or {}
        judge = (r.get("judge") or {}).get(e) or {}
        answers[e] = {"answer": a.get("answer", ""), "citations": trim_citations(a.get("citations")),
                      "telemetry": (r.get("telemetry") or {}).get(e), "reward": rewards[e],
                      "judge_score": judge.get("score"), "judge_rationale": judge.get("rationale") or judge.get("reason")}
    return {"question": r["question"], "dataset": r["dataset"], "split": r["split"],
            "category": r.get("benchmark_category", r.get("category")), "task": r.get("task", "text"),
            "classified_as": r.get("category"), "reference_answer": r.get("reference_answer"),
            "predicted_rewards": predicted, "router_engine": chosen,
            "router_reward": rewards.get(chosen) if chosen else None,
            "oracle_reward": max(v for v in rewards.values() if v is not None), "answers": answers}


def pick_showcase(rows, per_cell):
    """Per dataset x category, prefer questions where engines disagree (routing matters there)
    and readable question lengths; numeric FinQA questions form their own cell."""
    cells = {}
    for r in rows:
        cat = "numeric" if r.get("task") == "numeric" else r.get("benchmark_category", r.get("category"))
        cells.setdefault((r["dataset"], cat), []).append(r)
    picked = []
    for key in sorted(cells):
        def interest(r):
            rewards = [r.get("reward_" + e) or 0 for e in ENGINES]
            return (-(max(rewards) - min(rewards)), abs(len(r["question"]) - 140))
        picked += sorted((r for r in cells[key] if 40 <= len(r["question"]) <= 260), key=interest)[:per_cell]
    return picked


def hero_examples(rows, predict, limit=12):
    """Same shape as GET /api/rl/examples."""
    out = []
    for r in rows:
        predicted = predict(r.get("context")) if predict else None
        if not predicted or not (30 <= len(r["question"]) <= 170):
            continue
        rewards = {e: r.get("reward_" + e) for e in ENGINES}
        chosen = r.get("router_engine") or max(predicted, key=predicted.get)
        out.append({"question": r["question"], "category": r.get("benchmark_category", r["category"]), "task": r.get("task"),
                    "predicted_rewards": predicted, "chosen": chosen, "judged_rewards": rewards,
                    "router_reward": rewards.get(chosen), "oracle_reward": max(v for v in rewards.values() if v is not None)})
        if len(out) >= limit:
            break
    return out


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))] if values else None


def telemetry_summary(rows):
    out = {}
    for e in ENGINES:
        t = [(r.get("telemetry") or {}).get(e) or {} for r in rows]
        lat = [x["latency_ms"] for x in t if x.get("latency_ms") is not None]
        llm = [x["llm_ms"] for x in t if x.get("llm_ms") is not None]
        out[e] = {"n": len(lat), "latency_p50": percentile(lat, .5), "latency_p95": percentile(lat, .95),
                  "llm_share": (sum(llm) / sum(lat)) if lat and llm and sum(lat) else None,
                  "prompt_tokens": statistics.mean(x["prompt_tokens"] for x in t if x.get("prompt_tokens")) if any(x.get("prompt_tokens") for x in t) else None,
                  "completion_tokens": statistics.mean(x["completion_tokens"] for x in t if x.get("completion_tokens")) if any(x.get("completion_tokens") for x in t) else None}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "RL" / "data" / "runs" / "mixed")
    parser.add_argument("--output", type=Path, default=ROOT / "frontend" / "data" / "replay.json")
    parser.add_argument("--per-cell", type=int, default=2, help="showcase questions per dataset x category")
    args = parser.parse_args()
    rl, datasets = results_payload(args.run_dir)
    if not datasets:
        raise SystemExit(f"No results under {args.run_dir}/<dataset>/<split>")
    predict = predictor(args.run_dir)
    by_split = records_by_split(args.run_dir, datasets)
    shown = next(s for s in ("test", "dev", "train") if s in by_split)
    rows = by_split[shown]
    ingestion = {}
    for d in datasets:
        for s in SPLITS:
            for bundle, stats in ((load(args.run_dir / d / s / "summary.json") or {}).get("ingestion") or {}).items():
                ingestion[bundle] = {"dataset": d, "split": s, **stats}
    meta = next((s["summary"]["metadata"] for s in rl["splits"].values() if s["summary"]), {})
    payload = {
        "generated_at": time.strftime("%Y-%m-%d"),
        "showcase_split": shown, "held_out": shown != "train",
        "models": {k: meta.get(k) for k in ("answer_model", "extraction_model", "query_model", "judge_model",
                                            "question_model", "retrieval_version", "llm_backend")},
        "rl_results": rl,
        "examples": {"split": shown, "held_out": shown != "train", "examples": hero_examples(rows, predict)},
        "showcase": [showcase_entry(r, predict) for r in pick_showcase(rows, args.per_cell)],
        "telemetry": {s: telemetry_summary(r) for s, r in by_split.items()},
        "ingestion": ingestion,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1e6:.2f} MB): {len(payload['showcase'])} showcase "
          f"questions from {shown}, splits {list(by_split)}, datasets {datasets}")


if __name__ == "__main__":
    main()
