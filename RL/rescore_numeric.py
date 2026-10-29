"""Re-score saved numeric answers under the current numeric scorer, without new LLM calls.

Checkpoints are signed with the run metadata, which includes the metric version, so a scorer
change would otherwise discard every saved answer. This verifies each checkpoint was made under
the current LLM setup, re-scores numeric questions (text judge scores are kept), and re-signs it
under the new metric. Rerun run_public_benchmark.py afterwards to rebuild results and summaries."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "backend"))
from numeric_scoring import score_numeric
from run_public_benchmark import checkpoint_signature, metric_name


def rescore_split(manifest, checkpoint_dir):
    from app import engines
    metric = metric_name(manifest["dataset"])
    stats = {"checkpoints": 0, "numeric": 0, "already_current": 0, "flipped": []}
    for bundle in manifest["splits"][checkpoint_dir.parent.name]["bundles"]:
        for q in bundle["questions"]:
            path = checkpoint_dir / (hashlib.sha256(q["id"].encode()).hexdigest() + ".json")
            if not path.exists():
                continue
            saved = json.loads(path.read_text())
            record = saved["record"]
            if record["metadata"].get("metric") == metric:
                stats["already_current"] += 1
                continue
            if saved["signature"] != checkpoint_signature(record["metadata"], bundle):
                raise SystemExit(f"{q['id']}: checkpoint was made under a different LLM setup; "
                                 "dot-source scripts/use-ollama.ps1 with the training profile first.")
            record["metadata"] = {**record["metadata"], "metric": metric}
            if record.get("task") == "numeric":
                stats["numeric"] += 1
                for engine in engines.ENGINE_NAMES:
                    score = score_numeric(record["answers"][engine]["answer"], q)
                    old = record["reward_" + engine]
                    record["judge"][engine] = score
                    record["reward_" + engine] = score["score"] / 5
                    if record["reward_" + engine] != old:
                        stats["flipped"].append((q["id"], engine, old, record["reward_" + engine]))
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"signature": checkpoint_signature(record["metadata"], bundle), "record": record}, indent=2))
            temporary.replace(path)
            stats["checkpoints"] += 1
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data" / "shared_finqa")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "local-14b")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"], choices=("train", "dev", "test"))
    args = parser.parse_args()
    manifest = json.loads((args.dataset_dir / "manifest.json").read_text())
    for split in args.splits:
        checkpoint_dir = args.output_dir / split / "questions"
        if not checkpoint_dir.exists():
            print(f"{split}: no checkpoints")
            continue
        stats = rescore_split(manifest, checkpoint_dir)
        print(f"{split}: re-signed {stats['checkpoints']} checkpoints ({stats['numeric']} numeric, "
              f"{stats['already_current']} already current); {len(stats['flipped'])} engine scores changed")
        for qid, engine, old, new in stats["flipped"]:
            print(f"  {qid} {engine}: {old:.0f} -> {new:.0f}")


if __name__ == "__main__":
    main()
