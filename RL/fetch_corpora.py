"""
Fetches real corpora from SEC EDGAR's public, documented APIs:

  - https://www.sec.gov/files/company_tickers.json      (ticker -> CIK)
  - https://data.sec.gov/submissions/CIK##########.json  (a company's filing history)
  - https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{accession}-index.htm
    (per-filing document list, scraped for the EX-99 earnings-release exhibit)

IMPORTANT: this could not be tested against the live network in the sandbox
this was written in (SEC.gov isn't reachable from there) -- it's written
carefully against SEC's documented, stable API shapes, but real-world edge
cases (a ticker's most recent 8-K not having an earnings exhibit, a filing
using a different exhibit-numbering convention, transient rate-limit
errors) are likely to need a debugging pass on your machine. That's
expected and fine -- please report back what breaks.

SEC requires a descriptive User-Agent identifying who's making requests
(they will block generic ones) and asks for a modest request rate. SET
SEC_USER_AGENT BELOW to "YourName your-email@example.com" before running.

Usage:
    python fetch_corpora.py                 # fetch all 12 corpora
    python fetch_corpora.py --only semiconductors_2025
"""
from __future__ import annotations
import argparse
import html
import json
import re
import time
from pathlib import Path
from typing import List, Optional

import requests

SEC_USER_AGENT = "Ledger-Portfolio-Project samarthzaveri09@gmail.com"  # <-- EDIT THIS before running
REQUEST_DELAY_SECONDS = 0.25

CORPORA_DIR = Path(__file__).parent / "data" / "corpora"
MANIFEST_PATH = Path(__file__).parent / "data" / "corpus_manifest.json"

HEADERS = {"User-Agent": SEC_USER_AGENT}

# Each entry: (corpus_name, tickers, filings_per_ticker, category_label)
# category_label is just for our own bookkeeping/reporting -- it does NOT
# get fed to the bandit, which only ever sees the computed graph features.
CORPUS_SPECS = [
    # -- favorable: related companies or longitudinal, real cross-doc structure expected --
    ("semiconductors_2025", ["MXL", "QCOM", "SWKS"], 1, "favorable"),
    ("saas_2025", ["FIVN", "CRM", "NOW"], 1, "favorable"),
    ("maxlinear_longitudinal", ["MXL"], 4, "favorable"),
    ("homebuilders_2025", ["NVR", "DHI", "LEN"], 1, "favorable"),
    ("banks_2025", ["JPM", "BAC", "WFC"], 1, "favorable"),
    ("five9_longitudinal", ["FIVN"], 4, "favorable"),
    ("retail_2025", ["WMT", "TGT", "COST"], 1, "favorable"),
    ("airlines_2025", ["DAL", "UAL", "LUV"], 1, "favorable"),
    # -- control: deliberately unfavorable, no shared structure expected --
    ("control_single_aapl", ["AAPL"], 1, "control"),
    ("control_single_xom", ["XOM"], 1, "control"),
    ("control_mixed_unrelated_1", ["KO", "NFLX"], 1, "control"),
    ("control_mixed_unrelated_2", ["PFE", "CAT", "DIS"], 1, "control"),
]

_ticker_to_cik_cache: Optional[dict] = None


def _get(url: str, expect_json: bool = True):
    time.sleep(REQUEST_DELAY_SECONDS)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json() if expect_json else resp.text


def get_ticker_to_cik() -> dict:
    global _ticker_to_cik_cache
    if _ticker_to_cik_cache is not None:
        return _ticker_to_cik_cache
    data = _get("https://www.sec.gov/files/company_tickers.json")
    mapping = {entry["ticker"].upper(): entry["cik_str"] for entry in data.values()}
    _ticker_to_cik_cache = mapping
    return mapping


def get_recent_8k_accessions(cik: int, limit: int) -> List[str]:
    """Returns up to `limit` recent 8-K accession numbers (most recent first)."""
    padded_cik = f"{cik:010d}"
    data = _get(f"https://data.sec.gov/submissions/CIK{padded_cik}.json")
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    out = []
    for form, accession in zip(forms, accessions):
        if form == "8-K":
            out.append(accession)
        if len(out) >= limit:
            break
    return out


def find_earnings_exhibit_url(cik: int, accession: str) -> Optional[str]:
    """Scrapes a filing's index page for a document link that looks like an
    EX-99 earnings-release exhibit. Returns the full URL, or None."""
    accession_nodash = accession.replace("-", "")
    index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{accession}-index.htm"
    try:
        page = _get(index_url, expect_json=False)
    except requests.RequestException:
        return None

    hrefs = re.findall(r'href="([^"]+\.htm)"', page, re.IGNORECASE)
    # SEC exhibit filenames vary a lot in practice -- "exhibit991.htm",
    # "ex991earningsrelease.htm", "ex99-1.htm" all show up for the same
    # thing. Match "ex" or "exhibit" followed (with optional separators)
    # by "99", plus a couple of fallback keyword patterns.
    candidates = [h for h in hrefs if re.search(
        r"ex[\w\-.]*99|exhibit[\w\-.]*99|earnings|pressrelease|press-release", h, re.IGNORECASE
    )]
    if not candidates:
        return None
    chosen = candidates[0]
    if chosen.startswith("http"):
        return chosen
    if chosen.startswith("/"):
        return f"https://www.sec.gov{chosen}"
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{chosen}"


def html_to_text(raw_html: str) -> str:
    """Minimal, dependency-free HTML-to-text: strip script/style, drop tags,
    unescape entities, collapse whitespace. Good enough for SEC's fairly
    plain exhibit HTML -- not a general-purpose HTML parser."""
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw_html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<(br|p|div|tr|/tr|table|/table)[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def fetch_corpus(name: str, tickers: List[str], filings_per_ticker: int) -> List[dict]:
    ticker_map = get_ticker_to_cik()
    corpus_dir = CORPORA_DIR / name
    corpus_dir.mkdir(parents=True, exist_ok=True)

    fetched = []
    for ticker in tickers:
        cik = ticker_map.get(ticker.upper())
        if cik is None:
            print(f"  [{name}] WARNING: ticker {ticker} not found in SEC's ticker list, skipping")
            continue
        try:
            accessions = get_recent_8k_accessions(cik, limit=filings_per_ticker * 8)  # over-fetch, some 8-Ks won't have an earnings exhibit at all
        except requests.RequestException as e:
            print(f"  [{name}] WARNING: couldn't list filings for {ticker}: {e}")
            continue

        found = 0
        for accession in accessions:
            if found >= filings_per_ticker:
                break
            url = find_earnings_exhibit_url(cik, accession)
            if url is None:
                continue
            try:
                raw = _get(url, expect_json=False)
            except requests.RequestException as e:
                print(f"  [{name}] WARNING: couldn't fetch {url}: {e}")
                continue
            text = html_to_text(raw)
            if len(text) < 300:
                continue  # too short to be a real earnings release, likely wrong doc

            doc_id = f"{ticker.lower()}_{accession.replace('-', '')}"
            out_path = corpus_dir / f"{doc_id}.txt"
            out_path.write_text(f"SOURCE: {ticker} ({cik})\nURL: {url}\n\n{text}", encoding="utf-8")
            fetched.append({"ticker": ticker, "cik": cik, "accession": accession, "url": url, "path": str(out_path)})
            found += 1
            print(f"  [{name}] {ticker}: saved {out_path.name} ({len(text)} chars)")

        if found == 0:
            print(f"  [{name}] WARNING: found no earnings exhibit for {ticker} in its recent 8-Ks")

    return fetched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None, help="fetch just one corpus by name")
    args = parser.parse_args()

    if "your-email@example.com" in SEC_USER_AGENT:
        print("EDIT SEC_USER_AGENT at the top of this file with your real name/email before running "
              "(SEC blocks generic User-Agents).")
        return

    specs = CORPUS_SPECS if not args.only else [s for s in CORPUS_SPECS if s[0] == args.only]
    if not specs:
        print(f"No corpus spec named {args.only!r}")
        return

    manifest = {}
    for name, tickers, filings_per_ticker, category in specs:
        print(f"\n=== {name} ({category}) ===")
        fetched = fetch_corpus(name, tickers, filings_per_ticker)
        manifest[name] = {"category": category, "tickers": tickers, "documents": fetched}

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.exists() else {}
    existing.update(manifest)
    MANIFEST_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"\nManifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()