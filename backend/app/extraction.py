"""
Ingestion: chunk raw document text and extract (entity, relation, entity)
triples from each chunk using a local Ollama model with a strict JSON schema.
"""
from __future__ import annotations
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
    return ExtractionResult(
        doc_id=chunk["doc_id"], chunk_id=chunk["chunk_id"],
        entities=entities, triples=triples,
    )


def extract_document(doc_id: str, text: str) -> List[ExtractionResult]:
    chunks = chunk_text(text, doc_id)
    results = []
    for chunk in chunks:
        results.append(extract_from_chunk(chunk))
    return results