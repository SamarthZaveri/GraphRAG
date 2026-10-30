"""Build a cross-document benchmark from real SEC earnings releases.

FinQA bundles group single pages from unrelated companies, so their "global" and "conflict"
questions had to be forced across firms that share nothing. Here each corpus is a sector peer
group: three companies x their two most recent quarterly earnings releases (Form 8-K, Item 2.02,
Exhibit 99.1). The same company recurs across quarters and peers share metrics and themes,
which is the cross-document structure GraphRAG is designed for.

Splits are disjoint by sector, so no company appears in two splits. Questions come from the same
evidence-checked generator as shared_finqa (generate_benchmark.py), four categories balanced.

SEC fair-access policy requires a contact in the User-Agent: set LEDGER_SEC_USER_AGENT, e.g.
"Ledger research you@example.com". Downloads are cached under <output>/raw.

Usage: python RL/prepare_edgar.py --output-dir RL/data/edgar_xdoc [--download-only]"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "backend"))
from prepare_finqa import digest

VERSION = "edgar_earnings_v1"
SECTORS = {
    "train": {"airlines": ["DAL", "UAL", "LUV"], "restaurants": ["MCD", "SBUX", "CMG"],
              "payments": ["V", "MA", "PYPL"], "retail": ["WMT", "TGT", "COST"]},
    "dev": {"semiconductors": ["AMD", "NVDA", "MU"], "homebuilders": ["DHI", "LEN", "PHM"]},
    "test": {"cloud_software": ["CRM", "NOW", "WDAY"], "railroads": ["UNP", "CSX", "NSC"]},
}
RELEASES_PER_COMPANY = 2
MAX_DOC_CHARS = 80000  # releases end in long reconciliation tables; keep the narrative and main statements
REQUEST_INTERVAL = 0.2  # SEC allows 10 requests/second; stay well under it


class Edgar:
    def __init__(self, user_agent, cache_dir):
        if not user_agent or "@" not in user_agent:
            raise SystemExit("Set LEDGER_SEC_USER_AGENT to 'Name contact@email' (SEC fair-access policy).")
        self.headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
        self.cache = cache_dir
        self.last = 0.0

    def get(self, url, cache_name=None):
        path = self.cache / cache_name if cache_name else None
        if path and path.exists():
            return path.read_bytes()
        wait = REQUEST_INTERVAL - (time.monotonic() - self.last)
        if wait > 0:
            time.sleep(wait)
        for attempt in range(4):
            response = requests.get(url, headers=self.headers, timeout=(10, 60))
            self.last = time.monotonic()
            if response.status_code in (429, 503):
                time.sleep(2 ** attempt * 5)
                continue
            response.raise_for_status()
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(response.content)
            return response.content
        raise RuntimeError(f"SEC kept rate-limiting {url}")


def decode(content: bytes) -> str:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("cp1252", errors="replace")


def company_name(title):
    name = re.sub(r"\s*/[A-Z]{2,}/?\s*$", "", title.strip())  # "COSTCO WHOLESALE CORP /NEW"
    return " ".join(w if w.isupper() and len(w) <= 3 else w.capitalize() for w in name.split())


def earnings_releases(edgar, ticker, tickers):
    """The company's most recent Item 2.02 8-Ks that carry an HTML EX-99.1 release, newest first."""
    if ticker not in tickers:
        raise SystemExit(f"Ticker {ticker} not found in SEC company_tickers.json")
    cik = int(tickers[ticker]["cik_str"])
    submissions = json.loads(edgar.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json",
                                       f"{ticker}/submissions.json"))
    recent = submissions["filings"]["recent"]
    rows = [dict(zip(recent, values)) for values in zip(*recent.values())]
    found = []
    for row in rows:
        if row["form"] != "8-K" or "2.02" not in (row.get("items") or ""):
            continue
        accession = row["accessionNumber"]
        folder = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}"
        index = decode(edgar.get(f"{folder}/{accession}-index.html", f"{ticker}/{accession}-index.html"))
        candidates = []
        for cells_html in re.findall(r"<tr[^>]*>(.*?)</tr>", index, re.S | re.I):
            cells = [re.sub(r"<[^>]+>", "", c).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", cells_html, re.S | re.I)]
            href = re.search(r'href="([^"]+\.html?)"', cells_html, re.I)
            # Exhibit types vary by filer: EX-99.1, EX-99.01, EX-99. The press release is the first.
            if href and len(cells) >= 4 and re.match(r"EX-99(\.0?1)?$", cells[3].upper()):
                candidates.append((cells[3].upper() not in ("EX-99.1", "EX-99.01"),
                                   "https://www.sec.gov" + href.group(1).replace("/ix?doc=", "")))
        exhibit = min(candidates)[1] if candidates else None
        if exhibit:
            found.append({"ticker": ticker, "cik": cik, "company": company_name(submissions.get("name") or tickers[ticker]["title"]),
                          "filed": row["filingDate"], "accession": accession, "url": exhibit})
        if len(found) == RELEASES_PER_COMPANY:
            return found
    raise SystemExit(f"{ticker}: found {len(found)} earnings releases with an EX-99.1 exhibit; need {RELEASES_PER_COMPANY}")


def render(release, html):
    from app.html_text import html_to_text
    body = html_to_text(html)
    if len(body) > MAX_DOC_CHARS:
        cut = body.rfind("\n\n", 0, MAX_DOC_CHARS)
        body = body[:cut if cut > MAX_DOC_CHARS // 2 else MAX_DOC_CHARS]
    header = (f"COMPANY: {release['company']} ({release['ticker']})\n"
              f"DOCUMENT: Earnings release (Form 8-K Item 2.02, Exhibit 99.1), filed {release['filed']}\n"
              f"SOURCE: {release['url']}")
    return header + "\n\n" + body


def download(output, user_agent):
    edgar = Edgar(user_agent, output / "raw")
    tickers = {v["ticker"]: v for v in json.loads(edgar.get("https://www.sec.gov/files/company_tickers.json", "company_tickers.json")).values()}
    corpora = {}
    for split, sectors in SECTORS.items():
        for sector, symbols in sectors.items():
            bundle = f"{split}_{sector}"
            folder = output / "corpora" / bundle
            folder.mkdir(parents=True, exist_ok=True)
            docs = []
            for ticker in symbols:
                for release in earnings_releases(edgar, ticker, tickers):
                    doc_id = f"{ticker}_{release['filed']}"
                    path = folder / (doc_id + ".txt")
                    if not path.exists():
                        html = decode(edgar.get(release["url"], f"{ticker}/{release['accession']}.htm"))
                        path.write_text(render(release, html), encoding="utf-8")
                    docs.append({**release, "doc_id": doc_id, "chars": len(path.read_text(encoding="utf-8"))})
            corpora[bundle] = {"split": split, "sector": sector, "companies": symbols, "documents": docs}
            print(f"{bundle}: {len(docs)} releases, {sum(d['chars'] for d in docs):,} chars", flush=True)
    (output / "sources.json").write_text(json.dumps(corpora, indent=2))
    return corpora


def prepare(output, per_category, user_agent):
    from app import config
    from generate_benchmark import generate_benchmark_for_corpus, GENERATOR_VERSION
    corpora = download(output, user_agent)
    plan = {"version": VERSION, "generator": GENERATOR_VERSION, "per_category": per_category,
            "question_model": config.QUESTION_MODEL, "judge_model": config.JUDGE_MODEL,
            "qa_input_chars": config.QA_INPUT_CHARS, "backend": config.LLM_BACKEND,
            "sources": digest({b: [d["accession"] for d in c["documents"]] for b, c in corpora.items()})}
    lock = output / "source_plan.json"
    if lock.exists() and json.loads(lock.read_text()) != plan:
        raise SystemExit("Sources or generation settings changed; use a new --output-dir (existing work preserved).")
    lock.write_text(json.dumps(plan, indent=2))
    manifest = {"dataset": "edgar_xdoc", "version": VERSION, "source_revision": plan["sources"],
                "generation": plan, "prepared": False, "splits": {}}
    for split in SECTORS:
        bundles = []
        for bundle_id, corpus in corpora.items():
            if corpus["split"] != split:
                continue
            print(f"{split}: {bundle_id}", flush=True)
            doc_ids = [d["doc_id"] for d in corpus["documents"]]
            texts = [(output / "corpora" / bundle_id / (d + ".txt")).read_text(encoding="utf-8") for d in doc_ids]
            questions = generate_benchmark_for_corpus(texts, doc_ids, per_category,
                                                      checkpoint_path=output / "generated" / (bundle_id + ".partial.json"))
            saved = {"id": bundle_id, "doc_ids": doc_ids, "sector": corpus["sector"], "companies": corpus["companies"],
                     "questions": [{**q, "id": f"{bundle_id}:text:{q['id']}", "task": "text"} for q in questions]}
            saved["fingerprint"] = digest({"bundle": saved, "texts": texts})
            bundles.append(saved)
        manifest["splits"][split] = {"questions": sum(len(b["questions"]) for b in bundles), "bundles": bundles}
    manifest["prepared"] = True
    manifest["fingerprint"] = digest(manifest)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Prepared:", output / "manifest.json", {s: v["questions"] for s, v in manifest["splits"].items()})
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "edgar_xdoc")
    parser.add_argument("--per-category", type=int, choices=(3, 4, 5), default=4)
    parser.add_argument("--download-only", action="store_true", help="Fetch and convert filings without generating questions")
    args = parser.parse_args()
    user_agent = os.environ.get("LEDGER_SEC_USER_AGENT", "")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.download_only:
        download(args.output_dir, user_agent)
    else:
        prepare(args.output_dir, args.per_category, user_agent)


if __name__ == "__main__":
    main()
