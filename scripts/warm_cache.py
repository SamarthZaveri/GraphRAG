"""Pre-answer questions on an ingested corpus so a slow (CPU) deployment serves them instantly.

Runs the app in-process against LEDGER_DATA_DIR (no server needed), answering each question in
Auto mode and, with --compare, with all three engines. Answers land in the response cache
(<data>/response_cache.sqlite3) and the questions in <data>/suggested_questions.json, which the
UI shows as one-click questions. Questions come from the EDGAR benchmark's train split for the
corpora given (their documents must be the ones ingested in LEDGER_DATA_DIR).

    $env:LEDGER_DATA_DIR = "dist/demo_seed"
    python scripts/warm_cache.py --corpora train_airlines train_payments --per-corpus 6 --compare

Run it on the GPU machine with the same models as the deployment: the cache key includes the
models, the retrieval version and the corpus content, so answers made elsewhere are never served
for a different setup."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
CATEGORIES = ("local", "global", "multi_hop", "conflict")


def pick_questions(manifest, corpora, per_corpus):
    """Round-robin over categories so every question type is represented."""
    out = []
    for bundle in manifest["splits"]["train"]["bundles"]:
        if bundle["id"] not in corpora:
            continue
        by_cat = {c: [q for q in bundle["questions"] if q.get("category") == c] for c in CATEGORIES}
        picked = []
        while len(picked) < per_corpus and any(by_cat.values()):
            for c in CATEGORIES:
                if by_cat[c] and len(picked) < per_corpus:
                    q = by_cat[c].pop(0)
                    picked.append({"question": q["question"], "category": c, "corpus": bundle["id"]})
        out += picked
    missing = set(corpora) - {q["corpus"] for q in out}
    if missing:
        raise SystemExit(f"No train questions for {sorted(missing)}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpora", nargs="+", required=True)
    parser.add_argument("--per-corpus", type=int, default=6)
    parser.add_argument("--compare", action="store_true", help="also cache the three-engine comparison")
    parser.add_argument("--manifest", type=Path, default=ROOT / "RL" / "data" / "edgar_xdoc" / "manifest.json")
    args = parser.parse_args()
    from app import config, main as app, models, response_cache
    if response_cache.corpus_fingerprint() is None:
        raise SystemExit(f"No ingested corpus under {config.DATA_DIR}; set LEDGER_DATA_DIR to the seed directory")
    questions = pick_questions(json.loads(args.manifest.read_text(encoding="utf-8-sig")), set(args.corpora), args.per_corpus)
    started = time.perf_counter()
    for i, q in enumerate(questions, 1):
        t = time.perf_counter()
        r = app.query(models.QueryRequest(question=q["question"]))
        q["engine"] = r.engine_used
        if args.compare:
            app.query_compare(models.QueryRequest(question=q["question"]))
        q["cached"] = ["auto"] + (["compare"] if args.compare else [])
        print(f"[{i}/{len(questions)}] {time.perf_counter() - t:.0f}s {q['category']:9} -> {r.engine_used}: {q['question'][:70]}", flush=True)
    (config.DATA_DIR / "suggested_questions.json").write_text(json.dumps(questions, indent=2))
    print(f"Cached {len(questions)} questions in {(time.perf_counter() - started) / 60:.1f} min -> {config.RESPONSE_CACHE_PATH}")


if __name__ == "__main__":
    main()
