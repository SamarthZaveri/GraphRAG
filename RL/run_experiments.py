"""Evaluate three retrieval engines on a saved, balanced benchmark."""
from __future__ import annotations
import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent
# Resolve experiment storage before any backend import initializes its paths.
bootstrap = argparse.ArgumentParser(add_help=False)
bootstrap.add_argument("--output-dir", type=Path, default=ROOT / "data" / "three_engine")
bootstrap.add_argument("--state-dir", type=Path)
early, _ = bootstrap.parse_known_args()
OUTPUT_DIR = early.output_dir.resolve()
os.environ["LEDGER_DATA_DIR"] = str((early.state_dir or OUTPUT_DIR / "runtime").resolve())
sys.path.insert(0, str(ROOT.parent / "backend"))
from app import config, benchmark, engines, rgcn, vector_baseline
from app.main import _ingest_documents
from app.graph_store import GraphStore
from app.router_features import extract_features, build_query_context
from generate_benchmark import generate_benchmark_for_corpus, corpus_fingerprint, GENERATOR_VERSION, CATEGORIES


def load_benchmark(corpus_dir, benchmark_dir, target, allow_unbalanced=False, regenerate=False):
    paths = sorted(corpus_dir.glob("*.txt"))
    if not paths:
        raise ValueError(f"No .txt files in {corpus_dir}")
    texts = [p.read_text(encoding="utf-8", errors="ignore") for p in paths]
    ids = [p.stem for p in paths]
    fingerprint = corpus_fingerprint(texts, ids)
    path = benchmark_dir / (corpus_dir.name + ".json")
    if path.exists() and not regenerate:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("corpus_fingerprint") != fingerprint or saved.get("per_category") != target or saved.get("generator_version") != GENERATOR_VERSION:
            raise ValueError(f"Stale benchmark {path}; use --regenerate-questions explicitly")
        questions = saved["questions"]
    else:
        questions = generate_benchmark_for_corpus(texts, ids, target, allow_unbalanced)
        saved = {"corpus": corpus_dir.name, "corpus_fingerprint": fingerprint, "per_category": target,
                 "generator_version": GENERATOR_VERSION, "question_model": config.QUESTION_MODEL,
                 "reference_verifier_model": config.JUDGE_MODEL, "questions": questions}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(saved, indent=2, ensure_ascii=False), encoding="utf-8")
    counts = Counter(q["category"] for q in questions)
    if not allow_unbalanced and any(counts[c] != target for c in CATEGORIES):
        raise ValueError(f"Cached benchmark unbalanced: {dict(counts)}")
    if not questions:
        raise ValueError("Empty benchmark")
    return paths, saved


def run_one_corpus(corpus_dir, args):
    print(f"\nCorpus: {corpus_dir.name}")
    paths, saved = load_benchmark(corpus_dir, args.benchmark_dir, args.per_category,
                                  args.allow_unbalanced, args.regenerate_questions)
    if args.questions_only:
        return
    marker = config.DATA_DIR / "ingestion_manifest.json"
    expected = {"corpus_fingerprint": saved["corpus_fingerprint"], "extraction_model": config.EXTRACTION_MODEL,
                "retrieval_version": config.RETRIEVAL_VERSION}
    if args.reuse_index:
        if not marker.exists() or json.loads(marker.read_text()) != expected:
            raise ValueError("--reuse-index requires matching corpus, extraction model and retrieval version")
    else:
        _ingest_documents(paths, mode="replace")
        marker.write_text(json.dumps(expected), encoding="utf-8")
    store = GraphStore.load()
    features = extract_features(store)
    records, query_rows, failures = [], [], []
    meta = {**benchmark.metadata(), "corpus_fingerprint": saved["corpus_fingerprint"],
            "generator_version": saved["generator_version"], "benchmark_question_model": saved["question_model"],
            "benchmark_verifier_model": saved["reference_verifier_model"]}
    for q in saved["questions"]:
        print(f"  {q['id']} [{q['category']}]: {q['question']}")
        try:
            responses, scores = benchmark.evaluate_question(q)
        except Exception as error:
            print(f"  FAILED: {error}")
            failures.append({"question_id": q["id"], "reason": str(error)})
            continue
        rewards = {"reward_"+e: scores[e]["score"]/5.0 for e in engines.ENGINE_NAMES}
        query_rows.append({"corpus": corpus_dir.name, "question_id": q["id"], "question": q["question"],
                           "category": q["category"], "context": build_query_context(features, q["category"]).tolist(),
                           **rewards, "metadata": meta})
        records.append({**q, "answers": {e: r.model_dump() for e, r in responses.items()}, "judge": scores})
        print("  " + ", ".join(f"{e}={v['score']:.1f}/5" for e, v in scores.items()))
    if not query_rows:
        raise ValueError("No valid evaluated questions; no training rewards recorded")
    aggregate = {"corpus": corpus_dir.name, "num_docs": len(paths), "features": features.tolist(),
                 "num_questions": len(query_rows), "metadata": meta,
                 **{"reward_"+e: sum(r["reward_"+e] for r in query_rows)/len(query_rows) for e in engines.ENGINE_NAMES}}
    destination = OUTPUT_DIR / "corpora" / (corpus_dir.name + ".json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"aggregate": aggregate, "query_rows": query_rows,
        "results": records, "failed_questions": failures}, indent=2, ensure_ascii=False), encoding="utf-8")
    rebuild_training_files()
    if failures:
        raise ValueError(f"{len(failures)} questions failed; valid results saved, run is incomplete")


def rebuild_training_files():
    saved = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((OUTPUT_DIR / "corpora").glob("*.json"))]
    for name, rows in (("experiment_results.jsonl", [s["aggregate"] for s in saved]),
                       ("query_results.jsonl", [r for s in saved for r in s["query_rows"]])):
        (OUTPUT_DIR / name).write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, parents=[bootstrap])
    parser.add_argument("--corpus", help="Single corpus folder name")
    parser.add_argument("--corpora-dir", type=Path, default=ROOT / "data" / "corpora")
    parser.add_argument("--benchmark-dir", type=Path, default=ROOT / "data" / "balanced_benchmarks")
    parser.add_argument("--per-category", type=int, default=config.QA_PER_CATEGORY)
    parser.add_argument("--allow-unbalanced", action="store_true")
    parser.add_argument("--regenerate-questions", action="store_true")
    parser.add_argument("--questions-only", action="store_true")
    parser.add_argument("--reuse-index", action="store_true")
    args = parser.parse_args()
    if not args.corpora_dir.exists():
        parser.error("Corpus directory missing; run fetch_corpora.py or supply --corpora-dir")
    folders = sorted(p for p in args.corpora_dir.iterdir() if p.is_dir() and (not args.corpus or p.name == args.corpus))
    if not folders:
        parser.error("No matching corpora")
    if args.reuse_index and len(folders) != 1:
        parser.error("--reuse-index requires a single --corpus")
    failed = []
    for folder in folders:
        try:
            run_one_corpus(folder, args)
        except Exception:
            failed.append(folder.name)
            traceback.print_exc()
    if failed:
        raise SystemExit(f"Incomplete corpora: {', '.join(failed)}")
    print(f"Completed. Benchmarks: {args.benchmark_dir}; results: {OUTPUT_DIR}; state: {config.DATA_DIR}")


if __name__ == "__main__":
    main()
