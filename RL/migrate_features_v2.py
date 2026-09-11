"""
One-time migration: converts experiment_results.jsonl from the 6-dim
feature vector to the new 4-dim vector, WITHOUT re-running any corpus.

The old 6-dim order was:
  [cross_doc_entity_fraction, doc_pair_connectivity,
   cross_doc_subgraph_modularity, entity_recurrence_depth,
   num_docs_normalized, bias]

The new 4-dim order is:
  [cross_doc_entity_fraction, entity_recurrence_depth,
   num_docs_normalized, bias]

...which is exactly indices [0, 3, 4, 5] of the old vector. This is a
pure slice of already-computed, already-saved data -- the graph metrics
underlying these features don't need to be recomputed, so this script
does NOT touch Ollama, does NOT re-ingest anything, and takes under a
second to run.

Also deduplicates by corpus name, keeping the LAST occurrence of each
corpus in the file -- multiple re-runs of the same corpus (e.g.
five9_longitudinal was run 3 times while debugging) currently create
duplicate rows, all with the same corpus name. Keeping the last run
assumes the most recent run is the one you want to trust (e.g. after a
model swap or a bugfix); change KEEP = "first" below if you'd rather
keep the earliest run instead.

Prints a summary including which of the known 12 corpora are still
missing from the file, so you know exactly what (if anything) still
needs a real run.

Usage:
    python migrate_features_v2.py
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path

RESULTS_PATH = Path(__file__).parent / "data" / "experiment_results.jsonl"
BACKUP_PATH = Path(__file__).parent / "data" / "experiment_results_backup_6dim.jsonl"

# Indices into the OLD 6-dim vector that make up the NEW 4-dim vector.
OLD_TO_NEW_INDICES = [0, 3, 4, 5]

# Change to "first" if you'd rather keep each corpus's earliest run
# instead of its most recent one.
KEEP = "last"

KNOWN_12_CORPORA = {
    "airlines_2025", "banks_2025", "control_mixed_unrelated_1",
    "control_mixed_unrelated_2", "control_single_aapl", "control_single_xom",
    "five9_longitudinal", "homebuilders_2025", "maxlinear_longitudinal",
    "retail_2025", "saas_2025", "semiconductors_2025",
}


def main():
    if not RESULTS_PATH.exists():
        print(f"ERROR: {RESULTS_PATH} not found -- nothing to migrate.")
        return

    rows = [json.loads(line) for line in RESULTS_PATH.read_text().splitlines() if line.strip()]
    if not rows:
        print("ERROR: experiment_results.jsonl is empty -- nothing to migrate.")
        return

    bad_dim = [r["corpus"] for r in rows if len(r.get("features", [])) != 6]
    if bad_dim:
        print(f"ERROR: these rows are not 6-dim, can't slice them safely: {bad_dim}")
        print("If these are already 4-dim (already migrated), you don't need to run this again.")
        return

    shutil.copy(RESULTS_PATH, BACKUP_PATH)
    print(f"Backed up original 6-dim file to {BACKUP_PATH}")

    # Dedupe by corpus name, keeping first or last occurrence per KEEP setting.
    by_corpus: dict[str, dict] = {}
    for row in rows:
        name = row["corpus"]
        if KEEP == "last" or name not in by_corpus:
            by_corpus[name] = row

    migrated = []
    for name, row in by_corpus.items():
        old_features = row["features"]
        new_features = [old_features[i] for i in OLD_TO_NEW_INDICES]
        new_row = dict(row)
        new_row["features"] = new_features
        migrated.append(new_row)

    with open(RESULTS_PATH, "w") as f:
        for row in migrated:
            f.write(json.dumps(row) + "\n")

    print(f"\nMigrated {len(rows)} raw rows -> {len(migrated)} deduplicated 4-dim rows "
          f"(kept '{KEEP}' occurrence per corpus).")
    print(f"Written to {RESULTS_PATH}\n")

    have = set(by_corpus.keys())
    missing = KNOWN_12_CORPORA - have
    unexpected = have - KNOWN_12_CORPORA
    print(f"Corpora present ({len(have)}/12): {sorted(have)}")
    if missing:
        print(f"\nSTILL MISSING ({len(missing)}): {sorted(missing)}")
        print("These have no recorded row at all and need a real run if you want full coverage.")
    if unexpected:
        print(f"\nUnexpected corpus names not in the known-12 list: {sorted(unexpected)}")

    if len(have) < 10:
        print(f"\nNOTE: only {len(have)} distinct corpora recorded. train_bandit.py will still "
              f"run, but treat the result as directional given the small sample, same caveat as "
              f"before -- this migration doesn't fix sample size, only feature dimensionality.")


if __name__ == "__main__":
    main()