"""
For each corpus folder under RL/data/corpora/<name>/*.txt: ingest it through
the SAME backend pipeline the main app uses (extraction, graph construction,
community detection, R-GCN training), generate a benchmark for it, run both
GraphRAG and vector RAG against that benchmark, judge the answers, and
record one (context_features, reward_graphrag, reward_vector_rag) row per
corpus to RL/data/experiment_results.jsonl.

This deliberately reuses backend/app/* rather than reimplementing ingestion
-- if the real pipeline has a bug, we want the experiment to see it too,
not evaluate against a simplified stand-in.

Needs a running Ollama instance (same requirement as the main app). This is
the slow, real part: expect each corpus to take a few minutes depending on
your model and hardware.

Usage:
    python run_experiments.py                  # all corpora under data/corpora/
    python run_experiments.py --corpus semiconductors_2025   # just one
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
from app.community import build_community_summaries, save_summaries, load_modularity  # noqa: E402
from app import rgcn, vector_baseline, query_engine, benchmark as backend_benchmark  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from features import extract_features, FEATURE_NAMES  # noqa: E402
from generate_benchmark import generate_benchmark_for_corpus  # noqa: E402

CORPORA_DIR = Path(__file__).parent / "data" / "corpora"
RESULTS_PATH = Path(__file__).parent / "data" / "experiment_results.jsonl"


def ingest_corpus(corpus_dir: Path):
    """Runs the real ingestion pipeline for one corpus, in an isolated
    in-memory GraphStore + a corpus-scoped Chroma collection, so experiments
    don't collide with each other or with the main app's state."""
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

    store.save()  # query_engine/vector_baseline read graph state from disk
    summaries = build_community_summaries(store)
    save_summaries(summaries)
    rgcn_result = rgcn.train_and_save(store.graph)
    # NOTE: rgcn_result / val_auc is still computed and the R-GCN is still
    # trained and saved here -- the query engine / main app may still use
    # the trained embeddings for retrieval. It's only DROPPED from the
    # router's feature vector below (see features.py), because rgcn_val_auc
    # as a router feature was found to saturate uninformatively across
    # almost every corpus in the 12-corpus real run.

    return store, doc_texts, doc_ids, rgcn_result


def run_one_corpus(corpus_dir: Path) -> dict:
    print(f"\n=== {corpus_dir.name} ===")
    t0 = time.time()
    store, doc_texts, doc_ids, rgcn_result = ingest_corpus(corpus_dir)
    print(f"  ingested {len(doc_ids)} docs -> {store.graph.number_of_nodes()} nodes, "
          f"{store.graph.number_of_edges()} edges ({time.time()-t0:.0f}s)")

    modularity_info = load_modularity()
    val_auc = rgcn_result.get("val_auc") if rgcn_result else None
    # modularity_info / val_auc are still passed through for signature
    # compatibility but are IGNORED inside extract_features now -- see
    # features.py's revision note. Router features are recomputed there
    # directly from the graph's cross-document structure.
    features = extract_features(store, modularity_info.get("modularity"), val_auc)
    print(f"  features: {dict(zip(FEATURE_NAMES, features.round(3)))}")

    print("  generating benchmark questions...")
    questions = generate_benchmark_for_corpus(doc_texts, doc_ids)
    if not questions:
        print("  WARNING: no questions generated, skipping this corpus")
        return None
    print(f"  {len(questions)} questions generated")

    graphrag_scores, vector_scores = [], []
    for i, q in enumerate(questions, 1):
        graphrag_resp = query_engine.answer_question(q["question"], mode="auto")
        vector_resp = vector_baseline.answer_question(q["question"])
        verdict = backend_benchmark.judge(q["question"], q["reference_answer"],
                                           graphrag_resp.answer, vector_resp.answer)
        graphrag_scores.append(float(verdict.get("score_a", 0)))
        vector_scores.append(float(verdict.get("score_b", 0)))
        # DIAGNOSTIC (added while investigating the five9_longitudinal /
        # semiconductors_2025 low-score anomaly): print each question's
        # reference answer, both engines' scores, and the judge's
        # rationale. Previously only the aggregate average was visible,
        # which made it impossible to tell "both engines are genuinely
        # bad here" apart from "the reference answer itself is wrong,"
        # e.g. from truncated per-doc context in generate_benchmark.py.
        print(f"    Q{i}: {q['question'][:90]!r}")
        print(f"      reference: {q['reference_answer'][:150]!r}")
        print(f"      scores: graphrag={verdict.get('score_a')} vector={verdict.get('score_b')} "
              f"-- {verdict.get('rationale', '')[:180]}")

    reward_graphrag = sum(graphrag_scores) / len(graphrag_scores) / 5.0  # normalize 0-5 -> 0-1
    reward_vector = sum(vector_scores) / len(vector_scores) / 5.0
    print(f"  GraphRAG avg: {reward_graphrag*5:.2f}/5, Vector RAG avg: {reward_vector*5:.2f}/5 "
          f"({time.time()-t0:.0f}s total)")

    return {
        "corpus": corpus_dir.name,
        "num_docs": len(doc_ids),
        "features": features.tolist(),
        "reward_graphrag": reward_graphrag,
        "reward_vector_rag": reward_vector,
        "num_questions": len(questions),
    }


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

    if not corpus_dirs:
        print(f"No corpus folders found under {CORPORA_DIR}. Run fetch_corpora.py first.")
        sys.exit(1)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for corpus_dir in corpus_dirs:
        try:
            result = run_one_corpus(corpus_dir)
            if result:
                results.append(result)
        except Exception:
            print(f"  FAILED on {corpus_dir.name}:")
            traceback.print_exc()

    with open(RESULTS_PATH, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\n{len(results)} corpora completed, appended to {RESULTS_PATH}")


if __name__ == "__main__":
    main()