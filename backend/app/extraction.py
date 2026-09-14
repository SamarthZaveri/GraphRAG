"""
Ingestion: chunk raw document text and extract (entity, relation, entity)
triples from each chunk using a local Ollama model with a strict JSON schema.

DIAGNOSTIC NOTE (added while investigating slow ingestion): per-chunk
timing added to extract_from_chunk and a chunk-count summary added to
extract_document, no behavior change. Ingestion times have grown a lot
(15 min to 90+ min per corpus across different runs) and extraction is the
dominant cost, but WHY wasn't measured -- could be call count (many
chunks), per-call latency (slow generation), or Ollama serializing
concurrent requests despite EXTRACTION_CONCURRENCY=4 (see config.py's own
comment: this concurrency setting's real effect depends on Ollama's
OLLAMA_NUM_PARALLEL server setting, which has never been confirmed set).
This print output is what tells us which lever actually matters instead
of guessing.
"""
from __future__ import annotations
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

from . import config
from .ollama_client import chat_json
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
- For any numeric metric, capture the metric AND its fiscal period AND its value as distinct
  triples so figures stay comparable across periods and documents (don't just extract a bare
  number without tying it to which period/company it belongs to).
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
- Every triple must include a short "evidence" string (<= 25 words) paraphrased from the text
  supporting it — do not copy long verbatim spans, EXCEPT that stated percentages/comparisons
  (per the rule above) should be copied exactly, since altering a number while paraphrasing is
  worse than a slightly longer quote.
- Only extract what is actually supported by the text. Do not invent facts, and never invent or
  round numbers — copy figures exactly as written.
- Return ONLY valid JSON matching this schema, nothing else, no markdown fences, no preamble:

{
  "entities": [{"name": str, "type": str, "description": str}],
  "triples": [{"subject": str, "predicate": str, "object": str, "evidence": str}]
}
"""


def extract_from_chunk(chunk: dict) -> ExtractionResult:
    t0 = time.time()
    data = chat_json(
        config.EXTRACTION_MODEL, EXTRACTION_SYSTEM_PROMPT, chunk["text"], max_tokens=1500,
    )

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
    return ExtractionResult(
        doc_id=chunk["doc_id"], chunk_id=chunk["chunk_id"],
        entities=entities, triples=triples,
    )


def extract_document(doc_id: str, text: str) -> "tuple[list[dict], list[ExtractionResult]]":
    """Full per-document pipeline: strip out whitespace-aligned tables and
    extract them deterministically (table_parser.py, no LLM call, more
    reliable for numbers than a small model reading a flattened grid), then
    chunk and LLM-extract whatever prose is left, concurrently.

    Returns (chunk_records, extraction_results) -- chunk_records includes
    both the table blocks (as citable chunks) and the narrative chunks, so
    the caller can register all of them for citations/hybrid-grounding in
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

    narrative_chunks = chunk_text(remaining_text, doc_id)
    if narrative_chunks:
        print(f"  [extract_document] {doc_id}: {len(narrative_chunks)} narrative chunks to "
              f"process at concurrency={config.EXTRACTION_CONCURRENCY} "
              f"(model={config.EXTRACTION_MODEL})")
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=config.EXTRACTION_CONCURRENCY) as pool:
            llm_results = list(pool.map(extract_from_chunk, narrative_chunks))
        elapsed = time.time() - t0
        avg_per_chunk = elapsed / len(narrative_chunks)
        print(f"  [extract_document] {doc_id}: {len(narrative_chunks)} chunks finished in "
              f"{elapsed:.1f}s total ({avg_per_chunk:.1f}s/chunk average wall-clock at "
              f"concurrency={config.EXTRACTION_CONCURRENCY}). Compare this average against the "
              f"individual per-chunk times printed above: if wall-clock time is close to the SUM "
              f"of individual chunk times rather than roughly 1/{config.EXTRACTION_CONCURRENCY} "
              f"of it, Ollama is serializing requests despite the thread pool -- check "
              f"OLLAMA_NUM_PARALLEL on the server side.")
        results.extend(llm_results)

    return table_chunks + narrative_chunks, results