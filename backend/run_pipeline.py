#!/usr/bin/env python3
"""
Standalone CLI: ingest the bundled sample documents, build the graph +
community summaries, run the benchmark, and print a report. Useful for a
quick end-to-end sanity check without starting the FastAPI server / frontend.

Usage:
    ollama serve                     # if not already running
    ollama pull llama3.1             # once
    python run_pipeline.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app import config, ollama_client
from app.extraction import extract_document, chunk_text
from app.graph_store import reset_store
from app.community import build_community_summaries, save_summaries
from app import vector_baseline, benchmark


def main():
    if not ollama_client.is_available():
        print(f"ERROR: could not reach Ollama at {config.OLLAMA_HOST}.")
        print("Start it with `ollama serve` and make sure you've run "
              f"`ollama pull {config.EXTRACTION_MODEL}`.")
        sys.exit(1)

    print("=== Ledger: GraphRAG pipeline ===\n")
    paths = sorted(config.SAMPLE_DOCS_DIR.glob("*.txt"))
    print(f"Found {len(paths)} sample documents: {[p.name for p in paths]}\n")

    store = reset_store()
    vector_baseline.reset_collection()

    t0 = time.time()
    for path in paths:
        doc_id = path.stem
        text = path.read_text()
        print(f"-- Extracting entities/relations from {doc_id} ...")
        chunks = chunk_text(text, doc_id)
        results = extract_document(doc_id, text)
        store.ingest_document_chunks(doc_id, chunks, results)
        vector_baseline.ingest_document(doc_id, text)
        n_ent = sum(len(r.entities) for r in results)
        n_tri = sum(len(r.triples) for r in results)
        print(f"   {len(chunks)} chunks, {n_ent} entities, {n_tri} triples extracted")

    store.save()
    print(f"\nGraph built: {store.graph.number_of_nodes()} nodes, "
          f"{store.graph.number_of_edges()} edges ({time.time()-t0:.1f}s)\n")

    print("-- Running community detection + summarization ...")
    summaries = build_community_summaries(store)
    save_summaries(summaries)
    print(f"   {len(summaries)} communities summarized\n")
    for s in summaries:
        print(f"   [{s.title}] members: {', '.join(s.members[:6])}")
    print()

    print("-- Running benchmark (GraphRAG vs vector RAG) ...")
    summary = benchmark.run_benchmark()
    print(f"\n=== Benchmark results ===")
    print(f"GraphRAG avg score:   {summary.graphrag_avg:.2f} / 5")
    print(f"Vector RAG avg score: {summary.vector_rag_avg:.2f} / 5\n")
    for r in summary.results:
        print(f"[{r.category}] {r.question_id}: GraphRAG={r.graphrag_score} "
              f"VectorRAG={r.vector_rag_score}  -- {r.judge_rationale}")

    print(f"\nFull results written to {config.BENCHMARK_RESULTS_PATH}")


if __name__ == "__main__":
    main()
