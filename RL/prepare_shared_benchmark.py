"""Add balanced custom QA to the exact FinQA document bundles and splits."""
from __future__ import annotations
import argparse
import json
import shutil
import sys
from pathlib import Path
from prepare_finqa import digest
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "backend"))


def prepare(source, output, per_category=3):
    from app import config
    from generate_benchmark import generate_benchmark_for_corpus, GENERATOR_VERSION, COMPATIBLE_GENERATOR_VERSION
    original = json.loads((source / "manifest.json").read_text())
    plan = {"source_fingerprint": original["fingerprint"], "per_category": per_category,
            "generator": GENERATOR_VERSION, "question_model": config.QUESTION_MODEL, "judge_model": config.JUDGE_MODEL,
            "qa_input_chars": config.QA_INPUT_CHARS, "backend": config.LLM_BACKEND}
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "source_plan.json"
    if lock.exists():
        previous = json.loads(lock.read_text())
        if previous != plan:
            compatible = {**plan, "generator": COMPATIBLE_GENERATOR_VERSION}
            # Resume unfinished v4 work; completed experiments retain their
            # generation plan and require a new directory for changed prompts.
            if previous != compatible or (output / "manifest.json").exists():
                raise ValueError("Source/generation configuration changed; use a new output directory")
    lock.write_text(json.dumps(plan, indent=2))
    result = {"dataset": "shared_finqa", "version": "same_sources_numeric_text_v1", "source_revision": original["source_revision"],
              "source_fingerprint": original["fingerprint"], "generation": plan, "prepared": False, "splits": {}}
    for split, data in original["splits"].items():
        bundles = []
        for bundle in data["bundles"]:
            print(f"{split}: {bundle['id']}", flush=True)
            folder = output / "corpora" / bundle["id"]
            folder.mkdir(parents=True, exist_ok=True)
            paths = [source / "corpora" / bundle["id"] / (doc + ".txt") for doc in bundle["doc_ids"]]
            for path in paths:
                shutil.copyfile(path, folder / path.name)
            texts = [p.read_text(encoding="utf-8") for p in paths]
            cache = output / "generated" / (bundle["id"] + ".partial.json")
            custom = generate_benchmark_for_corpus(texts, bundle["doc_ids"], per_category, checkpoint_path=cache)
            numeric = [{**q, "task": "numeric"} for q in bundle["questions"]]
            text = [{**q, "id": bundle["id"] + ":text:" + q["id"], "task": "text"} for q in custom]
            saved = {"id": bundle["id"], "doc_ids": bundle["doc_ids"], "questions": numeric + text}
            saved["fingerprint"] = digest({"bundle": saved, "source_bundle": bundle["fingerprint"]})
            bundles.append(saved)
        result["splits"][split] = {"questions": sum(len(b["questions"]) for b in bundles), "bundles": bundles}
    result["prepared"] = True
    result["fingerprint"] = digest(result)
    (output / "manifest.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "data/public_finqa")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/shared_finqa")
    parser.add_argument("--per-category", type=int, choices=(3, 4, 5), default=3)
    args = parser.parse_args()
    prepare(args.source_dir, args.output_dir, args.per_category)


if __name__ == "__main__":
    main()
