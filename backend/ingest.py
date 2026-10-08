"""Ingest a folder of financial reports into the app's configured state."""
import argparse
from pathlib import Path
from app.main import _ingest_documents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--mode", choices=("replace", "add"), default="replace")
    args = parser.parse_args()
    paths = sorted(p for p in args.documents.glob("*") if p.is_file() and p.suffix.lower() in (".txt", ".pdf"))
    if not paths:
        parser.error("No .txt or .pdf reports found")
    response = _ingest_documents(paths, mode=args.mode)
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
