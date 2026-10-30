"""Collect benchmark training rewards or evaluate a frozen router, for one dataset and split.

Work runs per bundle in model-batched stages so a mixed-model profile swaps models twice per
bundle instead of twice per question:
  1. ingest the bundle's documents (extraction model);
  2. answer: classify each question once, then run all three engines in parallel on it,
     recording per-engine latency and tokens (query and answer models);
  3. score: exact scalar scoring for numeric questions, the blinded judge for text (judge model).
Answers and scores are checkpointed separately, so a judge or scorer change re-scores saved
answers without re-answering. Outputs: <output>/<dataset>/<split>/{results,summary}.json and,
for train, query_results.jsonl."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "backend"))
from prepare_finqa import digest

CATEGORIES = ("local", "global", "multi_hop", "conflict")


def metric_name(dataset):
    from numeric_scoring import VERSION
    return (f"shared_numeric_{VERSION.removeprefix('finqa_')}_text_judge_v1" if dataset == "shared_finqa"
            else "blinded_llm_judge_0_to_5")


def _endpoint():
    from app import config
    return {"endpoint": config.OLLAMA_HOST if config.LLM_BACKEND == "ollama" else config.API_BASE_URL,
            "thinking": config.NVIDIA_THINKING if config.LLM_BACKEND == "api" else False}


ANSWER_KEYS = ("retrieval_version", "llm_backend", "answer_model", "extraction_model", "query_model",
               "dataset", "dataset_fingerprint")


def answer_signature(meta, bundle):
    """Answers depend on retrieval, the answering/extraction/query models and the bundle."""
    return digest({"meta": {k: meta.get(k) for k in ANSWER_KEYS}, "bundle": bundle["fingerprint"], **_endpoint()})


def checkpoint_signature(meta, bundle):
    """A scored record additionally depends on the judge and the scoring metric."""
    return digest({"answers": answer_signature(meta, bundle), "judge_model": meta.get("judge_model"),
                   "metric": meta.get("metric")})


def ingestion_signature(meta, bundle):
    return digest({"meta": {k: meta.get(k) for k in ("retrieval_version", "llm_backend", "extraction_model")},
                   "bundle": bundle["fingerprint"], **_endpoint()})


def choose_frozen(policy, context):
    return max(policy.arms, key=lambda arm: policy.predicted_reward(arm, context))


def validate_policy(policy, manifest, meta):
    from app import engines, router_features
    if set(policy.arms) != set(engines.ENGINE_NAMES) or policy.context_dim != router_features.QUERY_CONTEXT_DIM:
        raise ValueError("Policy must use the current three engines and context dimensions")
    if policy.metadata.get("split") != "train":
        raise ValueError("Policy must be trained only on train splits")
    datasets = policy.metadata.get("datasets") or {policy.metadata.get("dataset"): policy.metadata.get("dataset_fingerprint")}
    # A dataset the policy trained on must be the same version (same split assignment); an unseen
    # dataset is a legitimate transfer evaluation.
    if manifest["dataset"] in datasets and datasets[manifest["dataset"]] != manifest["fingerprint"]:
        raise ValueError("Policy was trained on a different version of this dataset")
    for key in ("retrieval_version", "answer_model", "extraction_model", "judge_model", "llm_backend"):
        if policy.metadata.get(key) != meta[key]:
            raise ValueError(f"Policy mismatch for {key}; use the training model settings")


def run_metadata(manifest, split):
    from app import benchmark
    return {**benchmark.metadata(), "dataset": manifest["dataset"], "split": split,
            "dataset_fingerprint": manifest["fingerprint"], "source_revision": manifest["source_revision"],
            "metric": metric_name(manifest["dataset"])}


def _save(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def _load(path, signature):
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get("signature") == signature:
            return saved
    return None


def answer_all(q, features):
    """Classify once, then answer with every engine in parallel, each under its own trace."""
    from app import engines, query_engine, telemetry
    from app.router_features import build_query_context
    started = time.perf_counter()
    category, entities = query_engine.classify_query(q["question"])
    classify_ms = (time.perf_counter() - started) * 1000

    def run(engine):
        with telemetry.trace() as trace:
            response = engines.answer_question(engine, q["question"],
                                               classification=(category, entities) if engine == "graphrag" else None)
        return engine, response, trace.summary()

    with ThreadPoolExecutor(max_workers=len(engines.ENGINE_NAMES)) as pool:
        results = list(pool.map(telemetry.bind(run), engines.ENGINE_NAMES))
    return {"category": category, "entities": entities, "context": build_query_context(features, category).tolist(),
            "classify_ms": round(classify_ms, 1),
            "answers": {e: r.model_dump() for e, r, _ in results},
            "telemetry": {e: {k: m.get(k) for k in ("latency_ms", "llm_ms", "llm_calls", "prompt_tokens",
                                                    "completion_tokens", "cost_usd")} for e, _, m in results}}


def score(q, answered):
    from app import benchmark, engines
    answers = {e: answered["answers"][e]["answer"] for e in engines.ENGINE_NAMES}
    if q.get("task") == "numeric":
        from numeric_scoring import score_numeric
        return {e: score_numeric(a, q) for e, a in answers.items()}
    return benchmark.judge_answers(q["question"], q["reference_answer"], answers, q.get("evidence", []))


def ingest_bundle(args, bundle, meta):
    """Ingest a bundle unless the runtime index already holds exactly it; returns ingestion telemetry."""
    from app import config, telemetry
    from app.main import _ingest_documents
    marker = config.DATA_DIR / "public_ingestion_manifest.json"
    expected = {"signature": ingestion_signature(meta, bundle)}
    stats_path = config.DATA_DIR / "ingestion_stats" / (bundle["id"] + ".json")
    if marker.exists() and json.loads(marker.read_text()) == expected:
        return json.loads(stats_path.read_text()) if stats_path.exists() else None
    docs = [args.dataset_dir / "corpora" / bundle["id"] / (doc_id + ".txt") for doc_id in bundle["doc_ids"]]
    with telemetry.trace() as trace:
        result = _ingest_documents(docs, mode="replace")
    stats = {**{k: trace.summary().get(k) for k in ("latency_ms", "llm_ms", "llm_calls", "prompt_tokens", "completion_tokens", "cost_usd")},
             "documents": len(docs), "chars": sum(d.stat().st_size for d in docs),
             **{k: v for k in ("num_chunks", "num_entities", "num_triples", "num_communities")
                if isinstance(v := getattr(result, k, None), int)}}
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    _save(stats_path, stats)
    marker.write_text(json.dumps(expected))
    return stats


def summarize(records, meta, expected, failures, policy=None, ingestion=None):
    from app import engines
    names = engines.ENGINE_NAMES

    def mean(values):
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None

    def group(subset):
        return {"completed": len(subset),
                "engine_rewards": {e: mean(r["reward_" + e] for r in subset) for e in names},
                "router_reward": mean(r.get("router_reward") for r in subset) if policy else None}

    metrics = {e: mean(r["reward_" + e] for r in records) for e in names}
    if policy is not None:
        metrics["router"] = mean(r.get("router_reward") for r in records)
        metrics["oracle"] = mean(max(r["reward_" + e] for e in names) for r in records)
    return {"metadata": meta, "completed": len(records), "expected": expected,
            "complete": not failures and len(records) == expected,
            "category_counts": {c: sum(r["category"] == c for r in records) for c in CATEGORIES},
            "mean_normalized_judge_rewards": metrics,
            "by_benchmark_category": {c: group([r for r in records if r.get("benchmark_category") == c and r.get("task") != "numeric"])
                                      for c in CATEGORIES},
            "by_task": {t: group([r for r in records if r.get("task") == t]) for t in ("numeric", "text")},
            "latency_ms": {e: mean((r.get("telemetry") or {}).get(e, {}).get("latency_ms") for r in records) for e in names},
            "llm_calls": {e: mean((r.get("telemetry") or {}).get(e, {}).get("llm_calls") for r in records) for e in names},
            "ingestion": ingestion or {},
            "failed_questions": failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data" / "shared_finqa")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "runs" / "mixed")
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    parser.add_argument("--policy", type=Path, help="Required frozen train-only router for dev/test")
    args = parser.parse_args()
    if args.split != "train" and not args.policy:
        parser.error("dev/test require --policy; no training updates occur during evaluation")
    if args.split == "train" and args.policy:
        parser.error("--policy is for held-out evaluation only")
    os.environ["LEDGER_DATA_DIR"] = str((args.output_dir / "runtime").resolve())
    from app import engines
    from app.graph_store import GraphStore
    from app.router_features import extract_features
    from app.query_bandit import LinUCBBandit
    import tracking
    manifest = json.loads((args.dataset_dir / "manifest.json").read_text())
    if manifest.get("prepared") is False:
        raise ValueError("Benchmark questions are not generated yet; run the dataset's prepare script")
    meta = run_metadata(manifest, args.split)
    policy = LinUCBBandit.load(args.policy) if args.policy else None
    if policy is not None:
        validate_policy(policy, manifest, meta)
    policy_before = digest({"A": {a: v.tolist() for a, v in policy.A.items()}, "b": {a: v.tolist() for a, v in policy.b.items()}}) if policy else None
    destination = args.output_dir / manifest["dataset"] / args.split
    checkpoint_dir = destination / "questions"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    records, failures, ingestion = [], [], {}
    for bundle in manifest["splits"][args.split]["bundles"]:
        answer_sig, score_sig = answer_signature(meta, bundle), checkpoint_signature(meta, bundle)
        stem = {q["id"]: checkpoint_dir / hashlib.sha256(q["id"].encode()).hexdigest() for q in bundle["questions"]}
        final = {qid: saved["record"] for qid, path in stem.items() if (saved := _load(path.with_suffix(".json"), score_sig))}
        answered = {qid: saved for qid, path in stem.items() if qid not in final
                    and (saved := _load(path.with_suffix(".answers.json"), answer_sig))}
        print(f"{args.split} {bundle['id']}: {len(final)} scored, {len(answered)} answered, of {len(stem)}", flush=True)
        pending = [q for q in bundle["questions"] if q["id"] not in final and q["id"] not in answered]
        try:
            if pending:
                ingestion[bundle["id"]] = ingest_bundle(args, bundle, meta)
                features = extract_features(GraphStore.load())
                started = time.perf_counter()
                from app import query_engine
                warmed = query_engine.warm_embeddings(GraphStore.load())
                print(f"  embedded {warmed} graph facts and chunks in {time.perf_counter() - started:.0f}s", flush=True)
                for q in pending:
                    try:
                        answered[q["id"]] = {"signature": answer_sig, **answer_all(q, features)}
                        _save(stem[q["id"]].with_suffix(".answers.json"), answered[q["id"]])
                        print(f"  answered {q['id']}", flush=True)
                    except Exception as error:
                        failures.append({"question_id": q["id"], "stage": "answer", "error": str(error)})
                        print(f"  FAILED answer {q['id']}: {error}", flush=True)
            elif (args.output_dir / "runtime" / "ingestion_stats" / (bundle["id"] + ".json")).exists():
                ingestion[bundle["id"]] = json.loads((args.output_dir / "runtime" / "ingestion_stats" / (bundle["id"] + ".json")).read_text())
        except Exception as error:
            failures.extend({"question_id": q["id"], "stage": "ingest", "error": str(error)} for q in pending)
            print(f"  FAILED ingestion {bundle['id']}: {error}", flush=True)
        for q in bundle["questions"]:
            if q["id"] in final or q["id"] not in answered:
                continue
            try:
                a = answered[q["id"]]
                scores = score(q, a)
                final[q["id"]] = {"corpus": bundle["id"], "question_id": q["id"], "question": q["question"],
                                  "category": a["category"], "context": a["context"], "metadata": meta,
                                  "reference_answer": q["reference_answer"], "reference_program": q.get("reference_program"),
                                  "benchmark_category": q.get("category", a["category"]), "task": q.get("task", "text"),
                                  "answers": a["answers"], "telemetry": a["telemetry"], "classify_ms": a["classify_ms"],
                                  "judge": scores, **{"reward_" + e: scores[e]["score"] / 5 for e in engines.ENGINE_NAMES}}
                _save(stem[q["id"]].with_suffix(".json"), {"signature": score_sig, "record": final[q["id"]]})
                print(f"  scored {q['id']}", flush=True)
            except Exception as error:
                failures.append({"question_id": q["id"], "stage": "score", "error": str(error)})
                print(f"  FAILED score {q['id']}: {error}", flush=True)
        for q in bundle["questions"]:
            if q["id"] in final:
                record = dict(final[q["id"]])
                if policy is not None:
                    record["router_engine"] = choose_frozen(policy, record["context"])
                    record["router_reward"] = record["reward_" + record["router_engine"]]
                records.append(record)
    if policy is not None:
        after = digest({"A": {a: v.tolist() for a, v in policy.A.items()}, "b": {a: v.tolist() for a, v in policy.b.items()}})
        if after != policy_before:
            raise RuntimeError("Frozen policy changed during evaluation")
    summary = summarize(records, meta, manifest["splits"][args.split]["questions"], failures, policy, ingestion)
    (destination / "summary.json").write_text(json.dumps(summary, indent=2))
    (destination / "results.json").write_text(json.dumps(records, indent=2))
    # Only train rows are written to a trainable file; dev/test never become rewards input.
    if args.split == "train":
        (destination / "query_results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    tracking.log_evaluation(args.output_dir, summary, destination, policy_path=args.policy)
    print(json.dumps({k: summary[k] for k in ("completed", "expected", "complete", "mean_normalized_judge_rewards", "latency_ms")}, indent=2))
    if failures or not records:
        raise SystemExit("Incomplete run; successes checkpointed. Rerun the same command.")


if __name__ == "__main__":
    main()
