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


EXTRACTION_SYSTEM_PROMPT = """You are an information-extraction engine for a legal/financial \
knowledge-graph system called Ledger. Given a chunk of text from a contract or financial \
filing, extract:

1. entities: the parties, properties/assets, agreements, monetary amounts, dates, roles, and \
   locations that are named in the text.
2. triples: (subject, predicate, object) relationship facts stated or clearly implied by the \
   text, e.g. ("Sponsor", "shall not sell", "Sponsor SEDA Class B Shares"), \
   ("Acquiror", "is governed by", "Laws of the State of Delaware").

Rules:
- Use the exact names/labels as they appear in the text for subject/object when possible \
  (e.g. "the Sponsor", "SDCL EDGE Acquisition Corporation"), but normalize obvious defined-term \
  aliases to their fullest form when confident (e.g. if text defines "the Company" as \
  "Liberty Star Uranium and Metals Corp.", use the full name).
- entity "type" must be one of: Party, Property, Agreement, Money, Date, Obligation, Role, \
  Location, Other.
- Every triple must include a short "evidence" string (<= 25 words) paraphrased from the text \
  supporting it — do not copy long verbatim spans.
- Only extract what is actually supported by the text. Do not invent facts.
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
