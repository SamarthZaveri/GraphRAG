"""Collect public benchmark training rewards or evaluate a frozen router."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "backend"))
from prepare_finqa import digest


def metric_name(dataset):
    from numeric_scoring import VERSION
    return f"shared_numeric_{VERSION.removeprefix('finqa_')}_text_judge_v1" if dataset == "shared_finqa" else "blinded_llm_judge_0_to_5"


def checkpoint_signature(meta, bundle):
    """Checkpoints are reused only for the same run metadata, question bundle and LLM endpoint."""
    from app import config
    return digest({"meta": meta, "bundle": bundle["fingerprint"], "endpoint": config.OLLAMA_HOST if config.LLM_BACKEND == "ollama" else config.API_BASE_URL,
                   "thinking": config.NVIDIA_THINKING if config.LLM_BACKEND == "api" else False})


def choose_frozen(policy, context):
    return max(policy.arms, key=lambda arm: policy.predicted_reward(arm, context))


def validate_policy(policy, manifest, meta):
    from app import engines, router_features
    if set(policy.arms) != set(engines.ENGINE_NAMES) or policy.context_dim != router_features.QUERY_CONTEXT_DIM:
        raise ValueError("Policy must use the current three engines and context dimensions")
    dataset = manifest["dataset"]
    if policy.metadata.get("split") != "train":
        raise ValueError("Policy must be trained only on train splits")
    if (policy.metadata.get("dataset") != dataset or
          policy.metadata.get("dataset_fingerprint") != manifest["fingerprint"]):
        raise ValueError("Policy does not match this dataset manifest")
    for key in ("retrieval_version", "answer_model", "extraction_model", "judge_model", "llm_backend"):
        if policy.metadata.get(key) != meta[key]:
            raise ValueError(f"Policy mismatch for {key}; use the training model settings")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data" / "shared_finqa")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "local-14b")
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    parser.add_argument("--policy", type=Path, help="Required frozen train-only router for dev/test")
    args = parser.parse_args()
    if args.split != "train" and not args.policy:
        parser.error("dev/test require --policy; no training updates occur during evaluation")
    if args.split == "train" and args.policy:
        parser.error("--policy is for held-out evaluation only")
    os.environ["LEDGER_DATA_DIR"] = str((args.output_dir / "runtime").resolve())
    from app import config, benchmark, engines, query_engine
    from app.main import _ingest_documents
    from app.graph_store import GraphStore
    from app.router_features import extract_features, build_query_context
    from app.query_bandit import LinUCBBandit
    manifest = json.loads((args.dataset_dir / "manifest.json").read_text())
    if manifest.get("prepared") is False:
        raise ValueError("Shared QA is not generated yet; run prepare_shared_benchmark.py")
    meta = {**benchmark.metadata(), "dataset": manifest["dataset"], "split": args.split,
            "dataset_fingerprint": manifest["fingerprint"], "source_revision": manifest["source_revision"],
            "metric": metric_name(manifest["dataset"])}
    policy = LinUCBBandit.load(args.policy) if args.policy else None
    if policy is not None:
        validate_policy(policy, manifest, meta)
    policy_before = digest({"A": {a: v.tolist() for a, v in policy.A.items()}, "b": {a: v.tolist() for a, v in policy.b.items()}}) if policy else None
    records, failures = [], []
    checkpoint_dir = args.output_dir / args.split / "questions"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    marker = config.DATA_DIR / "public_ingestion_manifest.json"
    for bundle in manifest["splits"][args.split]["bundles"]:
        signature = checkpoint_signature(meta, bundle)
        paths = {q["id"]: checkpoint_dir / (hashlib.sha256(q["id"].encode()).hexdigest() + ".json") for q in bundle["questions"]}
        cached = {}
        for qid, path in paths.items():
            if path.exists():
                saved = json.loads(path.read_text())
                if saved.get("signature") == signature:
                    cached[qid] = saved["record"]
        print(f"{args.split} {bundle['id']}: {len(cached)}/{len(paths)} questions checkpointed", flush=True)
        try:
            if len(cached) < len(paths):
                expected = {"signature": signature}
                if not marker.exists() or json.loads(marker.read_text()) != expected:
                    docs = [args.dataset_dir / "corpora" / bundle["id"] / (doc_id + ".txt") for doc_id in bundle["doc_ids"]]
                    _ingest_documents(docs, mode="replace")
                    marker.write_text(json.dumps(expected))
                features = extract_features(GraphStore.load())
            for q in bundle["questions"]:
                try:
                    if q["id"] in cached:
                        record = cached[q["id"]]
                    else:
                        category, _ = query_engine.classify_query(q["question"])
                        if q.get("task") == "numeric":
                            from numeric_scoring import score_numeric
                            responses = {e: engines.answer_question(e, q["question"]) for e in engines.ENGINE_NAMES}
                            scores = {e: score_numeric(r.answer, q) for e, r in responses.items()}
                        else:
                            responses, scores = benchmark.evaluate_question(q)
                        context = build_query_context(features, category).tolist()
                        record = {"corpus": bundle["id"], "question_id": q["id"], "question": q["question"],
                                  "category": category, "context": context, "metadata": meta,
                                  "reference_answer": q["reference_answer"], "reference_program": q.get("reference_program"),
                                  "benchmark_category": q.get("category", category), "task": q.get("task", "text"),
                                  "answers": {e: r.model_dump() for e, r in responses.items()}, "judge": scores,
                                  **{"reward_" + e: scores[e]["score"] / 5 for e in engines.ENGINE_NAMES}}
                        path = paths[q["id"]]
                        temporary = path.with_suffix(".tmp")
                        temporary.write_text(json.dumps({"signature": signature, "record": record}, indent=2))
                        temporary.replace(path)
                    record = dict(record)
                    if policy is not None:
                        selected = choose_frozen(policy, record["context"])
                        record["router_engine"] = selected
                        record["router_reward"] = record["reward_" + selected]
                    records.append(record)
                    print("  " + q["id"] + " completed", flush=True)
                except Exception as error:
                    failures.append({"question_id": q["id"], "error": str(error)})
                    print(f"  FAILED {q['id']}: {error}", flush=True)
        except Exception as error:
            failures.extend({"question_id": q["id"], "error": str(error)} for q in bundle["questions"] if q["id"] not in cached)
            for cached_record in cached.values():
                record = dict(cached_record)
                if policy is not None:
                    selected = choose_frozen(policy, record["context"])
                    record["router_engine"] = selected
                    record["router_reward"] = record["reward_" + selected]
                records.append(record)
            print(f"  FAILED ingestion {bundle['id']}: {error}", flush=True)
    destination = args.output_dir / args.split
    destination.mkdir(parents=True, exist_ok=True)
    metrics = {e: sum(r["reward_" + e] for r in records) / len(records) if records else None for e in engines.ENGINE_NAMES}
    if policy is not None:
        after = digest({"A": {a: v.tolist() for a, v in policy.A.items()}, "b": {a: v.tolist() for a, v in policy.b.items()}})
        if after != policy_before:
            raise RuntimeError("Frozen policy changed during evaluation")
        routed = [r for r in records if "router_reward" in r]
        metrics["router"] = sum(r["router_reward"] for r in routed) / len(routed) if routed else None
        metrics["oracle"] = sum(max(r["reward_" + e] for e in engines.ENGINE_NAMES) for r in routed) / len(routed) if routed else None
    by_category = {}
    for category in ("local", "global", "multi_hop", "conflict"):
        subset = [r for r in records if r.get("benchmark_category", r["category"]) == category and r.get("task") != "numeric"]
        by_category[category] = {"completed": len(subset),
            "engine_rewards": {e: sum(r["reward_" + e] for r in subset) / len(subset) if subset else None for e in engines.ENGINE_NAMES},
            "router_reward": sum(r["router_reward"] for r in subset) / len(subset) if subset and policy else None}
    by_task = {}
    for task in ("numeric", "text"):
        subset = [r for r in records if r.get("task") == task]
        by_task[task] = {"completed": len(subset),
            "engine_rewards": {e: sum(r["reward_" + e] for r in subset) / len(subset) if subset else None for e in engines.ENGINE_NAMES},
            "router_reward": sum(r["router_reward"] for r in subset) / len(subset) if subset and policy else None}
    summary = {"metadata": meta, "completed": len(records), "expected": manifest["splits"][args.split]["questions"],
               "complete": not failures and len(records) == manifest["splits"][args.split]["questions"],
               "category_counts": {category: sum(r["category"] == category for r in records)
                                   for category in ("local", "global", "multi_hop", "conflict")},
               "mean_normalized_judge_rewards": metrics, "by_benchmark_category": by_category, "by_task": by_task, "failed_questions": failures}
    (destination / "summary.json").write_text(json.dumps(summary, indent=2))
    (destination / "results.json").write_text(json.dumps(records, indent=2))
    # Only train rows are written to a trainable file; dev/test never become rewards input.
    if args.split == "train":
        (destination / "query_results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    print(json.dumps(summary, indent=2))
    if failures or not records:
        raise SystemExit("Incomplete run; successes checkpointed. Rerun the same command.")


if __name__ == "__main__":
    main()
