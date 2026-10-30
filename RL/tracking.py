"""MLflow experiment tracking for benchmark evaluations, router training and analyses.

Every run records the full configuration that produced it (model per role, retrieval version,
scoring metric, dataset fingerprints, context sizes), its headline metrics and its output files,
so model profiles (e.g. 7B-answer vs 14B-answer) can be compared side by side in the MLflow UI:

    mlflow ui --backend-store-uri sqlite:///RL/mlflow.db

Tracking is optional and never fails a run: it is skipped when mlflow is not installed or
LEDGER_MLFLOW=0, and logging errors are reported and ignored. Runs are grouped by the run
directory name (the model profile) in the "run_group" tag."""
from __future__ import annotations
import json
import os
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPERIMENT = os.environ.get("LEDGER_MLFLOW_EXPERIMENT", "ledger-adaptive-rag")
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///" + (ROOT / "mlflow.db").as_posix())
ARTIFACT_ROOT = ROOT / "mlruns"


def _mlflow():
    if os.environ.get("LEDGER_MLFLOW", "1") == "0":
        return None
    try:
        import mlflow
    except ImportError:
        return None
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=ARTIFACT_ROOT.as_uri())
    mlflow.set_experiment(EXPERIMENT)
    return mlflow


def config_params():
    """Serving/benchmark configuration shared by every run type."""
    from app import config
    return {"retrieval_version": config.RETRIEVAL_VERSION, "llm_backend": config.LLM_BACKEND,
            "answer_model": config.ANSWER_MODEL, "extraction_model": config.EXTRACTION_MODEL,
            "query_model": config.QUERY_MODEL, "judge_model": config.JUDGE_MODEL,
            "question_model": config.QUESTION_MODEL, "extraction_concurrency": config.EXTRACTION_CONCURRENCY,
            "num_ctx": json.dumps(config.NUM_CTX_BY_MODEL) if config.NUM_CTX_BY_MODEL else config.OLLAMA_NUM_CTX,
            "graph_context_chars": config.GRAPH_CONTEXT_CHARS}


def _flat(prefix, values):
    return {f"{prefix}/{k}": float(v) for k, v in (values or {}).items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def _log(run_name, run_dir, params, metrics, artifacts, tags):
    mlflow = _mlflow()
    if mlflow is None:
        return
    try:
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({"run_group": Path(run_dir).name, **tags})
            mlflow.log_params({k: str(v)[:500] for k, v in params.items() if v is not None})
            mlflow.log_metrics({k: v for k, v in metrics.items() if v == v})  # drop NaN
            for path in artifacts:
                if Path(path).exists():
                    mlflow.log_artifact(str(path))
    except Exception:
        traceback.print_exc()
        print("MLflow logging failed; results on disk are unaffected.", flush=True)


def log_evaluation(run_dir, summary, destination, policy_path=None):
    meta = summary["metadata"]
    metrics = {**_flat("reward", summary["mean_normalized_judge_rewards"]),
               **_flat("latency_ms", summary.get("latency_ms")), **_flat("llm_calls", summary.get("llm_calls")),
               "questions": summary["completed"], "failures": len(summary["failed_questions"])}
    for task, group in (summary.get("by_task") or {}).items():
        metrics.update(_flat(f"{task}_reward", group.get("engine_rewards")))
    ingestion = [s for s in (summary.get("ingestion") or {}).values() if s]
    if ingestion:
        metrics["ingestion/llm_minutes"] = sum((s.get("llm_ms") or 0) for s in ingestion) / 60000
        metrics["ingestion/minutes"] = sum((s.get("latency_ms") or 0) for s in ingestion) / 60000
    _log(f"eval {meta['dataset']}/{meta['split']}", run_dir,
         {**config_params(), "dataset": meta["dataset"], "split": meta["split"], "metric": meta["metric"],
          "dataset_fingerprint": meta["dataset_fingerprint"][:16], "policy": str(policy_path) if policy_path else None},
         metrics, [destination / "summary.json"], {"stage": "evaluate", "dataset": meta["dataset"], "split": meta["split"]})


def log_router_training(run_dir, policy_path, metadata, settings, rows):
    _log("router train", run_dir,
         {**config_params(), **settings, "datasets": ",".join(sorted(metadata.get("datasets", {})))},
         {"training_rows": rows}, [policy_path, Path(policy_path).with_suffix(".training.json")],
         {"stage": "train_router"})


def log_analysis(run_dir, split, analysis, path):
    metrics = {}
    for name, block in [("all", analysis)] + list((analysis.get("by_dataset") or {}).items()):
        for policy, stats in (block.get("policies") or {}).items():
            if stats.get("mean") is not None:
                metrics[f"{name}/{policy}"] = stats["mean"]
        router = block.get("router") or {}
        for key in ("optimal_pick_rate", "mean_regret_vs_oracle"):
            if router.get(key) is not None:
                metrics[f"{name}/router_{key}"] = router[key]
        for engine, diff in (router.get("vs_engines") or {}).items():
            metrics[f"{name}/router_minus_{engine}"] = diff["mean_diff"]
    _log(f"analysis {split}", run_dir, {**config_params(), "split": split}, metrics, [path],
         {"stage": "analyze", "split": split})
