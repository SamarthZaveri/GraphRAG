"""Check the LLM judge against your own scores on a random sample of judged answers.

    python scripts/judge_audit.py sample --split dev --n 30     # writes RL/data/judge_audit/<split>.csv
    # open the CSV, fill human_score (0-5) for every row using the question, reference and answer
    python scripts/judge_audit.py score --split dev             # agreement report

Text answers only (numeric questions are scored exactly, not by the judge). The sample is
stratified across engines, and the judge's score is hidden in the CSV so it cannot anchor yours;
it is joined back by row id when scoring.
"""
from __future__ import annotations
import argparse
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINES = ("graphrag", "vector_rag", "hybrid_rag")


def load_rows(run, split):
    rows = []
    for path in sorted(run.glob(f"*/{split}/results.json")):
        if path.parent.parent.parent != run:
            continue
        for r in json.loads(path.read_text(encoding="utf-8")):
            if r.get("task") == "numeric":
                continue
            for e in ENGINES:
                judge = (r.get("judge") or {}).get(e) or {}
                if judge.get("score") is None:
                    continue
                rows.append({"id": f"{r['question_id']}|{e}", "dataset": path.parent.parent.name,
                             "category": r.get("benchmark_category"), "engine": e, "question": r["question"],
                             "reference_answer": r.get("reference_answer") or "",
                             "answer": r["answers"][e]["answer"], "judge_score": float(judge["score"])})
    return rows


def sample(args):
    rows = load_rows(args.run_dir, args.split)
    if not rows:
        raise SystemExit(f"No judged text answers under {args.run_dir}/<dataset>/{args.split}")
    rng = random.Random(args.seed)
    by_engine = {e: [r for r in rows if r["engine"] == e] for e in ENGINES}
    picked = []
    while len(picked) < args.n and any(by_engine.values()):
        for e in ENGINES:
            if by_engine[e] and len(picked) < args.n:
                picked.append(by_engine[e].pop(rng.randrange(len(by_engine[e]))))
    rng.shuffle(picked)
    out = args.out_dir / f"{args.split}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists (it may hold your scores); pass --force to replace it")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "dataset", "category", "question", "reference_answer", "answer", "human_score"])
        w.writeheader()
        for r in picked:  # engine and judge score withheld: blind to both
            w.writerow({k: r[k] for k in ("id", "dataset", "category", "question", "reference_answer", "answer")} | {"human_score": ""})
    print(f"Wrote {len(picked)} rows to {out}. Fill human_score (0-5), then run: score --split {args.split}")


def cohen_kappa(a, b):
    labels = sorted(set(a) | set(b))
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    expected = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def spearman(a, b):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    var = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return cov / var if var else float("nan")


def score(args):
    path = args.out_dir / f"{args.split}.csv"
    judged = {r["id"]: r for r in load_rows(args.run_dir, args.split)}
    pairs = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["human_score"].strip() and row["id"] in judged:
                pairs.append((float(row["human_score"]), judged[row["id"]]["judge_score"], judged[row["id"]]["engine"]))
    if len(pairs) < 10:
        raise SystemExit(f"Only {len(pairs)} scored rows in {path}; score at least 10 (30 recommended)")
    human, judge = [p[0] for p in pairs], [p[1] for p in pairs]
    passed = lambda s: s >= 3  # "acceptable answer" threshold on the 0-5 scale
    report = {
        "n": len(pairs),
        "exact_agreement": sum(h == j for h, j in zip(human, judge)) / len(pairs),
        "within_1": sum(abs(h - j) <= 1 for h, j in zip(human, judge)) / len(pairs),
        "mean_abs_diff": sum(abs(h - j) for h, j in zip(human, judge)) / len(pairs),
        "judge_minus_human": sum(j - h for h, j in zip(human, judge)) / len(pairs),
        "spearman": spearman(human, judge),
        "pass_fail_kappa": cohen_kappa([passed(h) for h in human], [passed(j) for j in judge]),
        "by_engine_judge_minus_human": {e: (sum(j - h for h, j, en in pairs if en == e) / n) if (n := sum(en == e for *_, en in pairs)) else None
                                        for e in ENGINES},
    }
    (args.out_dir / f"{args.split}_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("Positive judge_minus_human = the judge is more lenient than you. A by-engine gap means the judge favours an engine's style.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("sample", "score"))
    parser.add_argument("--split", default="dev")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "RL" / "data" / "runs" / "mixed")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "RL" / "data" / "judge_audit")
    args = parser.parse_args()
    sample(args) if args.action == "sample" else score(args)


if __name__ == "__main__":
    main()
