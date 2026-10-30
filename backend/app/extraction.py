"""Extract financial entities and relationships through the configured LLM.

Tables are parsed separately; narrative chunks use bounded output-token retries
and report per-chunk progress without accepting truncated model responses.
"""
from __future__ import annotations
import time
import hashlib
import json
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import List

from . import config, telemetry
from .llm_client import chat_json, OutputTruncatedError, LLMError
from .models import ExtractionResult, Entity, Triple


def chunk_text(text: str, doc_id: str, size: int = config.CHUNK_SIZE_CHARS,
               overlap: int = config.CHUNK_OVERLAP_CHARS) -> List[dict]:
    """Simple character-window chunker that tries to break on paragraph/sentence
    boundaries so entities and their context stay together."""
    text = text.strip()
    chunks = []
    start = 0
    idx = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            # try to break on a paragraph or sentence boundary near `end`
            window = text[start:end]
            for sep in ["\n\n", ". ", "\n"]:
                pos = window.rfind(sep)
                if pos > size * 0.5:
                    end = start + pos + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({
                "chunk_id": f"{doc_id}::chunk{idx}",
                "doc_id": doc_id,
                "text": chunk,
            })
            idx += 1
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


EXTRACTION_SYSTEM_PROMPT = """You are an information-extraction engine for a financial-report \
knowledge-graph system called Ledger. Given a chunk of text from an earnings release, 10-K/10-Q,
or other financial filing, extract:

1. entities: companies, financial metrics (revenue, gross margin, net income, EBITDA, EPS,
   opex, debt, cash, etc.), fiscal periods (e.g. "Q2 2025", "six months ended June 30, 2025"),
   business segments, officers/executives, and named risk factors.
2. triples: (subject, predicate, object) relationship facts stated or clearly implied by the
   text, e.g. ("MaxLinear net revenue", "was in period", "Q2 2025"),
   ("MaxLinear net revenue Q2 2025", "has value", "$108.8 million"),
   ("MaxLinear net revenue", "increased sequentially vs", "MaxLinear net revenue Q1 2025"),
   ("Five9", "reports segment", "Enterprise AI revenue").

Rules:
- Tables were already parsed into facts separately. Focus on what prose adds: figures stated in
  sentences, guidance and outlook, drivers and explanations of changes, events (acquisitions,
  debt, buybacks, appointments), segment commentary and risks.
- Put the company and period in the metric name so a value is unambiguous: subject
  "MaxLinear net revenue Q2 2025", predicate "has value", object "$108.8 million". Never extract
  a bare number without the company and period it belongs to.
- If the text states a percentage change, growth rate, or comparison between periods (e.g. "up
  13% sequentially", "margin decreased to 21.5% from 23.6%"), capture that AS ITS OWN TRIPLE with
  the stated percentage copied verbatim into the evidence field — e.g.
  ("MaxLinear net revenue Q2 2025", "changed vs Q1 2025 by", "up 13% sequentially") with evidence
  "net revenue was $108.8 million, up 13% sequentially". Do this even though you're also
  capturing the raw values separately — the pre-computed comparison is a fact worth keeping in
  its own right, since a downstream reader should never have to redo that arithmetic themselves.
- Use the exact company/segment names as they appear in the text (e.g. "MaxLinear, Inc.",
  "Five9, Inc.", "NVR, Inc."), and keep metric names consistent within a document (e.g. always
  "net revenue" not sometimes "revenue" if the text uses "net revenue").
- entity "type" must be one of: Company, Metric, Period, Segment, Person, RiskFactor, Guidance,
  Other.
- Every triple must include a short "evidence" string (<= 15 words) paraphrased from the text
  supporting it — do not copy long verbatim spans, EXCEPT that stated percentages/comparisons
  (per the rule above) should be copied exactly, since altering a number while paraphrasing is
  worse than a slightly longer quote.
- Only extract what is actually supported by the text. Do not invent facts, and never invent or
  round numbers — copy figures exactly as written.
- Descriptions are at most 8 words. Avoid duplicate entities or triples, and skip legal
  boilerplate, contact details and generic definitions.
- Return ONLY valid JSON matching this schema, nothing else, no markdown fences, no preamble:

{
  "entities": [{"name": str, "type": str, "description": str}],
  "triples": [{"subject": str, "predicate": str, "object": str, "evidence": str}]
}
"""


_BOILERPLATE = re.compile(
    r"forward-looking statements|safe harbor|private securities litigation reform act|"
    r"investor relations|media contact|^contacts?:|conference call|webcast|"
    r"non-gaap financial measures? (?:are|is|should)|not (?:been )?prepared in accordance with gaap|"
    r"undue reliance|risk factors.{0,40}annual report on form 10-k", re.I | re.M)


def strip_boilerplate(text: str) -> str:
    """Drop legal and logistics paragraphs (safe-harbor language, contacts, webcast details,
    non-GAAP disclaimers) before LLM extraction: they carry no facts but cost output tokens.
    Only extraction input is filtered; vector and hybrid retrieval index the full text."""
    paragraphs = re.split(r"\n\s*\n", text)
    return "\n\n".join(p for p in paragraphs if not _BOILERPLATE.search(p))


MIN_SPLIT_CHARS = 600


def _halves(text: str) -> "tuple[str, str]":
    """Split near the middle on a paragraph, sentence or word boundary."""
    middle = len(text) // 2
    for sep in ("\n\n", ". ", "\n", " "):
        left, right = text.rfind(sep, 0, middle + 200), text.find(sep, middle - 200)
        candidates = [p for p in (left, right) if p > len(text) // 4 and p < len(text) * 3 // 4]
        if candidates:
            cut = min(candidates, key=lambda p: abs(p - middle)) + len(sep)
            return text[:cut].strip(), text[cut:].strip()
    return text[:middle], text[middle:]


def _extract_json(text: str, label: str, depth: int = 0) -> dict:
    """Entities and triples for one passage. A truncated response from a ~1.7k-char chunk is a
    generation loop rather than a lack of room, so the passage is split in half (cheaper and more
    reliable than a larger budget). A short passage gets one larger budget, then is skipped with
    a warning instead of aborting ingestion of the whole corpus."""
    budget = config.EXTRACTION_MAX_TOKENS
    invalid_attempts = 0
    prompt = EXTRACTION_SYSTEM_PROMPT
    while True:
        try:
            data = chat_json(config.EXTRACTION_MODEL, prompt, text, max_tokens=budget)
            if isinstance(data.get("entities"), list) and isinstance(data.get("triples"), list):
                return data
            invalid_attempts += 1
            if invalid_attempts >= 3:
                raise LLMError(f"Invalid extraction JSON for {label} after 3 attempts; expected entities and triples lists.")
            prompt = EXTRACTION_SYSTEM_PROMPT + '\nYour previous response lacked the required lists. Return {"entities": [], "triples": []} if the passage has no supported facts. Otherwise return both lists using the schema. No prose or null values.'
            print(f"    [extract] {label}: invalid JSON/schema; retry {invalid_attempts+1}/3", flush=True)
        except OutputTruncatedError:
            if len(text) >= MIN_SPLIT_CHARS and depth < 2:
                print(f"    [extract] {label}: output truncated at {budget} tokens; splitting the passage", flush=True)
                parts = [_extract_json(part, f"{label}/{i + 1}", depth + 1) for i, part in enumerate(_halves(text))]
                return {"entities": [e for p in parts for e in p["entities"]],
                        "triples": [t for p in parts for t in p["triples"]]}
            if budget < config.EXTRACTION_TOKEN_LIMIT:
                budget = min(budget * 2, config.EXTRACTION_TOKEN_LIMIT)
                print(f"    [extract] {label}: output truncated; retrying with {budget} tokens", flush=True)
                continue
            print(f"    [extract] WARNING {label}: still truncated at {budget} tokens; skipped "
                  f"({len(text)} chars; tables and the vector indexes still cover it)", flush=True)
            return {"entities": [], "triples": []}


def extract_from_chunk(chunk: dict) -> ExtractionResult:
    t0 = time.time()
    fingerprint = hashlib.sha256(json.dumps({
        "version": 2, "chunk": chunk, "model": config.EXTRACTION_MODEL,
        "backend": config.LLM_BACKEND,
        "endpoint": config.API_BASE_URL if config.LLM_BACKEND == "api" else config.OLLAMA_HOST,
        "prompt": EXTRACTION_SYSTEM_PROMPT, "thinking": config.NVIDIA_THINKING,
    }, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    cache_path = config.DATA_DIR / "extraction_cache" / (fingerprint + ".json")
    if cache_path.exists():
        try:
            result = ExtractionResult.model_validate_json(cache_path.read_text(encoding="utf-8"))
            if result.doc_id == chunk["doc_id"] and result.chunk_id == chunk["chunk_id"]:
                print(f"    [extract] {chunk['chunk_id']}: reused checkpoint", flush=True)
                return result
        except (ValueError, OSError):
            pass  # Recompute corrupt checkpoints; never trust unvalidated cache data.
    data = _extract_json(chunk["text"], chunk["chunk_id"])
    entities = [
        Entity(name=e["name"], type=e.get("type", "Other"), description=e.get("description"),
               source_docs=[chunk["doc_id"]])
        for e in data.get("entities", []) if isinstance(e, dict) and e.get("name")
    ]
    triples = [
        Triple(
            subject=t["subject"], predicate=t["predicate"], object=t["object"],
            doc_id=chunk["doc_id"], chunk_id=chunk["chunk_id"],
            evidence=t.get("evidence", ""),
        )
        for t in data.get("triples", [])
        if isinstance(t, dict) and t.get("subject") and t.get("predicate") and t.get("object")
    ]
    elapsed = time.time() - t0
    print(f"    [extract] {chunk['chunk_id']} ({len(chunk['text'])} chars in) -> "
          f"{len(entities)} entities, {len(triples)} triples ({elapsed:.1f}s)")
    result = ExtractionResult(
        doc_id=chunk["doc_id"], chunk_id=chunk["chunk_id"],
        entities=entities, triples=triples,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix("." + uuid.uuid4().hex[:8] + ".tmp")  # short: Windows paths cap at 260 chars
    try:
        temporary.write_text(result.model_dump_json(), encoding="utf-8")
        temporary.replace(cache_path)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def extract_document(doc_id: str, text: str) -> "tuple[list[dict], list[ExtractionResult]]":
    """Full per-document pipeline: strip out whitespace-aligned tables and
    extract them deterministically (table_parser.py, no LLM call, more
    reliable for numbers than a small model reading a flattened grid), then
    chunk and LLM-extract whatever prose is left, concurrently.

    Returns (chunk_records, extraction_results) -- chunk_records includes
    both the table blocks (as citable chunks) and the narrative chunks, so
    the caller can register all of them for citations/source-grounding in
    one place.
    """
    from . import table_parser

    table_entities, table_triples, remaining_text = table_parser.detect_and_extract_tables(text, doc_id)
    table_chunks = table_parser.table_chunk_texts(text, doc_id)

    results: List[ExtractionResult] = []
    if table_entities or table_triples:
        results.append(ExtractionResult(
            doc_id=doc_id, chunk_id=f"{doc_id}::tables",
            entities=table_entities, triples=table_triples,
        ))

    narrative_chunks = chunk_text(strip_boilerplate(remaining_text), doc_id)
    if narrative_chunks:
        print(f"  [extract_document] {doc_id}: {len(narrative_chunks)} narrative chunks to "
              f"process at concurrency={config.EXTRACTION_CONCURRENCY} "
              f"(model={config.EXTRACTION_MODEL})")
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=config.EXTRACTION_CONCURRENCY) as pool:
            llm_results = list(pool.map(telemetry.bind(extract_from_chunk), narrative_chunks))
        elapsed = time.time() - t0
        avg_per_chunk = elapsed / len(narrative_chunks)
        print(f"  [extract_document] {doc_id}: {len(narrative_chunks)} chunks finished in "
              f"{elapsed:.1f}s total ({avg_per_chunk:.1f}s/chunk at "
              f"concurrency={config.EXTRACTION_CONCURRENCY}, backend={config.LLM_BACKEND}).", flush=True)
        results.extend(llm_results)

    return table_chunks + narrative_chunks, results