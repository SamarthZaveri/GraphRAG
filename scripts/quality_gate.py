"""CI quality gate on the newest frozen release (releases/<tag>/): fails the build when the
committed results regress or no longer match their manifest.

    python scripts/quality_gate.py                       # newest release by MANIFEST frozen_at
    python scripts/quality_gate.py --release releases/v1-router --tolerance 0.02

Checks, on the held-out split (test, else dev):
  1. integrity: every file listed in MANIFEST.json exists with the recorded sha256
  2. completeness: every split's run completed with no failed questions
  3. the router is not significantly worse than any single engine (paired bootstrap)
  4. the router's mean reward is within --tolerance of the best single engine
  5. retrieval: the best engine's document hit rate is at least --min-hit-rate
Passes with a notice when no release has been frozen yet. Standard library only."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINES = ("graphrag", "vector_rag", "hybrid_rag")


def newest_release():
    manifests = sorted(ROOT.glob("releases/*/MANIFEST.json"),
                       key=lambda p: json.loads(p.read_text()).get("frozen_at", ""))
    return manifests[-1].parent if manifests else None


def check(release, tolerance, min_hit_rate):
    manifest = json.loads((release / "MANIFEST.json").read_text())
    failures, notes = [], []
    for rel, digest in manifest.get("files", {}).items():
        path = release / rel
        if not path.exists():
            failures.append(f"missing file {rel}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            failures.append(f"{rel} changed since the release was frozen")
    for split, complete in (manifest.get("complete") or {}).items():
        summary = json.loads((release / split / "summary.json").read_text())
        if complete is not True or summary.get("failed_questions"):
            failures.append(f"{split}: incomplete run ({len(summary.get('failed_questions') or [])} failed questions)")
    split = next((s for s in ("test", "dev") if (release / s / "analysis.json").exists()), None)
    if split is None:
        failures.append("no held-out (test or dev) analysis in the release")
        return failures, notes
    analysis = json.loads((release / split / "analysis.json").read_text())
    policies = analysis.get("policies", {})
    engines = {e: policies[f"always_{e}"]["mean"] for e in ENGINES if f"always_{e}" in policies}
    best = max(engines, key=engines.get)
    router = analysis.get("router") or {}
    if "router" not in policies:
        failures.append(f"{split}: no router evaluation")
    else:
        mean = policies["router"]["mean"]
        notes.append(f"{split}: router {mean:.3f} vs best engine {best} {engines[best]:.3f} (n={policies['router']['n']})")
        if mean < engines[best] - tolerance:
            failures.append(f"{split}: router {mean:.3f} is more than {tolerance} below {best} {engines[best]:.3f}")
        for engine, diff in (router.get("vs_engines") or {}).items():
            if diff and diff["significant"] and diff["mean_diff"] < 0:
                failures.append(f"{split}: router significantly worse than {engine} "
                                f"({diff['mean_diff']:+.3f}, 95% CI {diff['ci95']})")
    hit = ((analysis.get("retrieval") or {}).get(best) or {}).get("doc_hit_rate")
    if hit is not None:
        notes.append(f"{split}: {best} document hit rate {hit:.3f}")
        if hit < min_hit_rate:
            failures.append(f"{split}: {best} document hit rate {hit:.3f} below {min_hit_rate}")
    return failures, notes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--min-hit-rate", type=float, default=0.5)
    args = parser.parse_args()
    release = args.release or newest_release()
    if release is None:
        print("No frozen release yet (releases/<tag>/MANIFEST.json); quality gate skipped.")
        return
    failures, notes = check(release, args.tolerance, args.min_hit_rate)
    print(f"Quality gate: {release.relative_to(ROOT) if release.is_relative_to(ROOT) else release}")
    for note in notes:
        print("  " + note)
    if failures:
        print("FAILED:\n" + "\n".join("  - " + f for f in failures))
        sys.exit(1)
    print("passed")


if __name__ == "__main__":
    main()
