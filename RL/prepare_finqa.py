"""Prepare public FinQA source documents and existing QA; no LLM generation."""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
VERSION = "finqa_rag_v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def report_id(example):
    # Keep all pages of a company/year report together across the selected splits.
    return "/".join(example["id"].split("/")[:2])


def render_document(example):
    # Allowlist only source fields; never ingest qa, programs or retrieved gold inputs.
    lines = ["SOURCE: " + example["id"].rsplit("-", 1)[0]]
    lines.extend(example.get("pre_text", []))
    table = example.get("table", [])
    if table:
        header = table[0]
        lines.append("TABLE HEADERS: " + " | ".join(map(str, header)))
        for row in table[1:]:
            cells = [f"{header[i] if i < len(header) else 'column ' + str(i)}: {cell}"
                     for i, cell in enumerate(row[1:], 1)]
            lines.append("TABLE ROW " + str(row[0]) + " | " + " ; ".join(cells))
    lines.extend(example.get("post_text", []))
    return "\n\n".join(lines)


def prepare(raw, output, limits, seed=0, group_size=8, revision="local"):
    if group_size < 1 or any(limit < 1 for limit in limits.values()):
        raise ValueError("Limits and group size must be positive")
    seen_reports, seen_questions = set(), set()
    manifest = {"dataset": "finqa", "version": VERSION, "source_revision": revision,
                "seed": seed, "group_size": group_size, "splits": {}}
    for split in ("train", "dev", "test"):
        candidates = list(raw[split])
        random.Random(seed).shuffle(candidates)
        chosen, excluded = [], 0
        for example in candidates:
            qa = example.get("qa", {})
            if report_id(example) in seen_reports or example["id"] in seen_questions:
                excluded += 1
                continue
            if not qa.get("question") or "exe_ans" not in qa:
                continue
            chosen.append(example)
            if len(chosen) == limits[split]:
                break
        if not chosen:
            raise ValueError(f"No labeled, report-disjoint {split} examples available")
        seen_reports.update(report_id(e) for e in chosen)
        seen_questions.update(e["id"] for e in chosen)
        documents = {}
        for example in chosen:
            text = render_document(example)
            doc_id = "doc_" + digest(text)[:20]
            documents.setdefault(doc_id, {"text": text, "examples": []})["examples"].append(example)
        bundles = []
        items = sorted(documents.items())
        for offset in range(0, len(items), group_size):
            bundle = f"{split}_{offset // group_size:04d}"
            folder = output / "corpora" / bundle
            folder.mkdir(parents=True, exist_ok=True)
            questions, doc_ids = [], []
            for doc_id, document in items[offset:offset + group_size]:
                (folder / (doc_id + ".txt")).write_text(document["text"], encoding="utf-8")
                doc_ids.append(doc_id)
                for example in document["examples"]:
                    qa = example["qa"]
                    questions.append({"id": example["id"], "question": qa["question"],
                        "reference_answer": str(qa["exe_ans"]), "human_answer": qa.get("answer"),
                        "reference_program": qa.get("program"), "source_doc_id": doc_id,
                        "report_id": report_id(example),
                        "evidence": [{"doc_id": doc_id, "quote": value}
                                     for value in qa.get("gold_inds", {}).values() if isinstance(value, str)]})
            saved = {"id": bundle, "doc_ids": doc_ids, "questions": questions}
            saved["fingerprint"] = digest({"saved": saved, "texts": [documents[d]["text"] for d in doc_ids]})
            bundles.append(saved)
        manifest["splits"][split] = {"questions": len(chosen), "excluded_overlap": excluded, "bundles": bundles}
        print(f"{split}: {len(chosen)} questions, {len(documents)} documents, {len(bundles)} bundles", flush=True)
    manifest["fingerprint"] = digest(manifest)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "public_finqa")
    parser.add_argument("--train-limit", type=int, default=40)
    parser.add_argument("--dev-limit", type=int, default=20)
    parser.add_argument("--test-limit", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--revision", default="main", help="Upstream Git commit or branch; resolved to a commit on first download")
    args = parser.parse_args()
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    lock = raw_dir / "revision.txt"
    if lock.exists():
        revision = lock.read_text().strip()
        if args.revision != "main" and args.revision != revision:
            parser.error("Existing download is pinned to a different revision; choose another output directory")
    else:
        response = requests.get("https://api.github.com/repos/czyssrs/FinQA/commits/" + args.revision, timeout=(10, 30))
        response.raise_for_status()
        revision = response.json()["sha"]
        lock.write_text(revision)
    raw = {}
    for split in ("train", "dev", "test"):
        path = raw_dir / (split + ".json")
        if not path.exists():
            response = requests.get(f"https://raw.githubusercontent.com/czyssrs/FinQA/{revision}/dataset/{split}.json", timeout=(10, 60))
            response.raise_for_status()
            data = response.json()
            path.write_text(json.dumps(data), encoding="utf-8")
        raw[split] = json.loads(path.read_text(encoding="utf-8"))
    prepare(raw, args.output_dir, {s: getattr(args, s + "_limit") for s in raw}, args.seed, args.group_size, revision)
    print("Prepared:", args.output_dir / "manifest.json")


if __name__ == "__main__":
    main()
