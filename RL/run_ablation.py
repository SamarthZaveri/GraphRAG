"""
ABLATION: graph-only GraphRAG vs vector RAG, same questions for both.

Why this exists: GraphRAG's local search has always pulled in the same
vector-similarity chunks vector RAG uses, as a "hybrid grounding" safety
net (see query_engine.py). That means every GraphRAG-vs-vector-RAG number
collected so far in this project wasn't really testing graph retrieval
against vector retrieval -- it was testing (graph + vector) against
(vector alone), which structurally favors GraphRAG regardless of whether
its actual graph traversal is any good. This script isolates that: it
calls query_engine.answer_question(..., use_hybrid=False) so GraphRAG
answers using ONLY graph-derived facts and chunks, nothing else, and
compares that against vector RAG (unchanged -- it was already vector-only)
on the exact same question set.

This is a diagnostic, not a replacement for run_experiments.py / the
query-level router training data -- it doesn't write to
experiment_results.jsonl or query_results.jsonl, so it can't accidentally
corrupt the router's training data with numbers from a deliberately
handicapped GraphRAG. Results go to a separate file, ablation_results.jsonl.

Same questions for both engines is achieved the straightforward way: one
generate_benchmark_for_corpus() call per corpus produces one question set,
and BOTH engines answer every question in it -- there's no separate
question generation per engine.

Ingestion for each corpus should be fast on a re-run of documents already
processed in a prior session, thanks to the extraction cache (see
extraction.py) -- only the answer/judge stage needs to run fresh, since
that's what this ablation actually changes.

Usage:
    python run_ablation.py                       # all corpora
    python run_ablation.py --corpus five9_longitudinal
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from app import config as backend_config, ollama_client  # noqa: E402
from app.extraction import extract_document  # noqa: E402
from app.graph_store import GraphStore  # noqa: E402
from app.community import build_community_summaries, save_summaries  # noqa: E402
from app import rgcn, vector_baseline, query_engine, benchmark as backend_benchmark  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from generate_benchmark import generate_benchmark_for_corpus  # noqa: E402

CORPORA_DIR = Path(__file__).parent / "data" / "corpora"
RESULTS_PATH = Path(__file__).parent / "data" / "ablation_results.jsonl"


def ingest_corpus(corpus_dir: Path):
    """Same ingestion as run_experiments.py's ingest_corpus -- kept as a
    separate copy here rather than imported, so this diagnostic script has
    zero risk of an import-time side effect on the main pipeline's state."""
    doc_paths = sorted(corpus_dir.glob("*.txt"))
    if not doc_paths:
        raise FileNotFoundError(f"No .txt documents found in {corpus_dir}")

    store = GraphStore()
    vector_baseline.reset_collection()

    doc_texts, doc_ids = [], []
    for path in doc_paths:
        doc_id = path.stem
        text = path.read_text(errors="ignore")
        doc_texts.append(text)
        doc_ids.append(doc_id)

        chunks, extraction_results = extract_document(doc_id, text)
        store.ingest_document_chunks(doc_id, chunks, extraction_results)
        vector_baseline.ingest_document(doc_id, text)

    store.save()
    summaries = build_community_summaries(store)
    save_summaries(summaries)
    try:
        rgcn.train_and_save(store.graph)
    except Exception:
        traceback.print_exc()  # not load-bearing for this ablation

    return store, doc_texts, doc_ids


def run_one_corpus(corpus_dir: Path) -> list[dict]:
    print(f"\n=== {corpus_dir.name} ===")
    t0 = time.time()
    store, doc_texts, doc_ids = ingest_corpus(corpus_dir)
    print(f"  ingested {len(doc_ids)} docs -> {store.graph.number_of_nodes()} nodes, "
          f"{store.graph.number_of_edges()} edges ({time.time()-t0:.0f}s -- should be fast if "
          f"these documents were processed before, thanks to the extraction cache)")

    print("  generating benchmark questions...")
    questions = generate_benchmark_for_corpus(doc_texts, doc_ids)
    if not questions:
        print("  WARNING: no questions generated, skipping this corpus")
        return []
    print(f"  {len(questions)} questions generated (SAME set used for both engines below)")

    rows = []
    for i, q in enumerate(questions, 1):
        # The only change from run_experiments.py: use_hybrid=False.
        graphrag_resp = query_engine.answer_question(q["question"], mode="auto", use_hybrid=False)
        vector_resp = vector_baseline.answer_question(q["question"])
        verdict = backend_benchmark.judge(q["question"], q["reference_answer"],
                                           graphrag_resp.answer, vector_resp.answer)
        score_graphrag = float(verdict.get("score_a", 0))
        score_vector = float(verdict.get("score_b", 0))
        print(f"    Q{i}: {q['question'][:90]!r}")
        print(f"      scores: graphrag_pure={score_graphrag} vector={score_vector} "
              f"-- {verdict.get('rationale', '')[:150]}")

        rows.append({
            "corpus": corpus_dir.name,
            "question": q["question"],
            "category": q.get("category", "local"),
            "reference_answer": q["reference_answer"],
            "graphrag_pure_score": score_graphrag / 5.0,
            "vector_rag_score": score_vector / 5.0,
        })

    avg_graphrag = sum(r["graphrag_pure_score"] for r in rows) / len(rows) * 5.0
    avg_vector = sum(r["vector_rag_score"] for r in rows) / len(rows) * 5.0
    print(f"  GraphRAG (graph-only) avg: {avg_graphrag:.2f}/5, Vector RAG avg: {avg_vector:.2f}/5 "
          f"({time.time()-t0:.0f}s total)")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=None, help="Run just one corpus by folder name")
    args = parser.parse_args()

    if not ollama_client.is_available():
        print(f"ERROR: can't reach Ollama at {backend_config.OLLAMA_HOST}. Start it first.")
        sys.exit(1)

    corpus_dirs = sorted(d for d in CORPORA_DIR.iterdir() if d.is_dir())
    if args.corpus:
        corpus_dirs = [d for d in corpus_dirs if d.name == args.corpus]
        if not corpus_dirs:
            print(f"No corpus folder named {args.corpus!r} under {CORPORA_DIR}")
            sys.exit(1)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for corpus_dir in corpus_dirs:
        try:
            all_rows.extend(run_one_corpus(corpus_dir))
        except Exception:
            print(f"  FAILED on {corpus_dir.name}:")
            traceback.print_exc()

    with open(RESULTS_PATH, "a") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")

    print(f"\n{len(all_rows)} ablation rows appended to {RESULTS_PATH}")
    print("This file is separate from experiment_results.jsonl / query_results.jsonl -- "
          "it does NOT feed the router's training data.")


if __name__ == "__main__":
    main()