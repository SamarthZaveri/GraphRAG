"""Rescore frozen answers with a different judge without regenerating QA."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from app import benchmark, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Choose a separate output to preserve the original scores")
    saved = json.loads(args.input.read_text(encoding="utf-8"))
    rescored, failures = [], []
    for q in saved["results"]:
        try:
            verdict = benchmark.judge_answers(q["question"], q["reference_answer"],
                       {engine: response["answer"] for engine, response in q["answers"].items()}, q.get("evidence", []))
            rescored.append({"question_id": q["id"], "category": q["category"], "judge": verdict})
        except benchmark.JudgeError as error:
            failures.append({"question_id": q["id"], "reason": str(error)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"source": str(args.input), "judge_model": config.JUDGE_MODEL,
            "results": rescored, "failed_questions": failures}, indent=2), encoding="utf-8")
    print(f"Saved {len(rescored)} rescored questions; {len(failures)} failures")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
