"""Freeze a finished run as a versioned, committed release: releases/<tag>/.

Working data under RL/data is gitignored (checkpoints, indexes, raw filings). A release keeps what
a reader needs to trust and reproduce the headline numbers:

    policy.json                         the trained router (deployable as query_bandit.json)
    <split>/{summary,analysis}.json     combined results per split
    <dataset>/<split>/{summary,analysis,results}.json
    MANIFEST.json                       git commit, models, retrieval version, metric versions,
                                        dataset fingerprints, file hashes

    python scripts/freeze_release.py --tag v1-router
    python scripts/freeze_release.py --tag v1-router --run-dir RL/data/runs/mixed --allow-dirty

Refuses to overwrite an existing tag, and (unless --allow-dirty) a working tree with uncommitted
code changes, since the commit hash would not describe the code that produced the numbers.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "dev", "test")


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True, help="release name, e.g. v1-router")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "RL" / "data" / "runs" / "mixed")
    parser.add_argument("--allow-dirty", action="store_true", help="freeze even with uncommitted code changes")
    args = parser.parse_args()
    out = ROOT / "releases" / args.tag
    if out.exists():
        raise SystemExit(f"{out} exists; choose a new tag (releases are immutable)")
    dirty = [l for l in git("status", "--porcelain").splitlines() if not l[3:].startswith(("releases/", "frontend/data/"))]
    if dirty and not args.allow_dirty:
        raise SystemExit("Uncommitted changes:\n" + "\n".join(dirty) + "\nCommit first, or pass --allow-dirty.")
    run = args.run_dir
    if not (run / "policy.json").exists():
        raise SystemExit(f"No policy.json under {run}")

    copied = []
    def copy(src, rel):
        if src.exists():
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            copied.append(rel)
    copy(run / "policy.json", "policy.json")
    for split in SPLITS:
        for name in ("summary.json", "analysis.json"):
            copy(run / split / name, f"{split}/{name}")
    datasets = sorted(p.name for p in run.iterdir() if p.is_dir() and p.name not in SPLITS and p.name != "runtime")
    for d in datasets:
        for split in SPLITS:
            for name in ("summary.json", "analysis.json", "results.json"):
                copy(run / d / split / name, f"{d}/{split}/{name}")

    summaries = {s: json.loads((run / s / "summary.json").read_text()) for s in SPLITS if (run / s / "summary.json").exists()}
    per_dataset = {d: {s: json.loads((run / d / s / "summary.json").read_text())
                       for s in SPLITS if (run / d / s / "summary.json").exists()} for d in datasets}
    meta = next(iter(next(iter(per_dataset.values())).values()))["metadata"] if per_dataset else {}
    policy = json.loads((run / "policy.json").read_text())
    manifest = {
        "tag": args.tag, "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": git("rev-parse", "HEAD"), "dirty": bool(dirty), "run_dir": str(run.relative_to(ROOT)),
        "models": {k: meta.get(k) for k in ("answer_model", "extraction_model", "query_model", "question_model",
                                            "judge_model", "llm_backend")},
        "retrieval_version": meta.get("retrieval_version"),
        "metrics": {d: next(iter(s.values()))["metadata"].get("metric") for d, s in per_dataset.items() if s},
        "dataset_fingerprints": {d: next(iter(s.values()))["metadata"].get("dataset_fingerprint") for d, s in per_dataset.items() if s},
        "policy": {k: policy.get(k) for k in ("arms", "alpha", "context_dim")} | {"metadata": policy.get("metadata")},
        "questions": {d: {s: v.get("completed") for s, v in sp.items()} for d, sp in per_dataset.items()},
        "complete": {s: v.get("complete") for s, v in summaries.items()},
        "files": {rel: sha256(out / rel) for rel in sorted(copied)},
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    # Point the registry's "champion" alias at the frozen policy, with its held-out numbers.
    sys.path.insert(0, str(ROOT / "RL"))
    sys.path.insert(0, str(ROOT / "backend"))
    import tracking
    held_out = next((s for s in ("test", "dev") if (run / s / "analysis.json").exists()), None)
    metrics = {}
    if held_out:
        analysis = json.loads((run / held_out / "analysis.json").read_text())
        metrics = {f"{held_out}/{p}": v["mean"] for p, v in analysis.get("policies", {}).items() if v.get("mean") is not None}
    version = tracking.register_policy(run / "policy.json", {**(policy.get("metadata") or {}), "release": args.tag},
                                       metrics, alias="champion")
    if version:
        print(f"MLflow model {tracking.REGISTERED_MODEL} version {version} is now the 'champion'")
    print(f"Froze {len(copied)} files to {out}\n  commit {manifest['git_commit'][:10]}{' (dirty)' if dirty else ''}"
          f" | retrieval {manifest['retrieval_version']} | questions {manifest['questions']}")
    print(f"Tag it: git add releases/{args.tag} && git commit -m \"Release {args.tag}\" && git tag {args.tag}")


if __name__ == "__main__":
    main()
