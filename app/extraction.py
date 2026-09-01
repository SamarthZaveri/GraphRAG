"""
Ingestion: chunk raw document text and extract (entity, relation, entity)
triples from each chunk using the Anthropic API with a strict JSON schema.
"""
from __future__ import annotations
import json
import re
from typing import List

from anthropic import Anthropic

from . import config
from .models import ExtractionResult, Entity, Triple

_client: Anthropic | None = None


def get_client() -> Anthropic:
    global _client
    if _client is None:
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it before starting the server."
            )
        _client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _client


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
- Return ONLY valid JSON matching this schema, nothing else, no markdown fences:

{
  "entities": [{"name": str, "type": str, "description": str}],
  "triples": [{"subject": str, "predicate": str, "object": str, "evidence": str}]
}
"""


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def extract_from_chunk(chunk: dict) -> ExtractionResult:
    client = get_client()
    resp = client.messages.create(
        model=config.EXTRACTION_MODEL,
        max_tokens=2000,
        system=EXTRACTION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": chunk["text"]}],
    )
    raw = "".join(b.text for b in resp.content if b.type == "text")
    raw = _strip_json_fences(raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # best-effort recovery: grab the first {...} block
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group(0)) if match else {"entities": [], "triples": []}

    entities = [
        Entity(name=e["name"], type=e.get("type", "Other"), description=e.get("description"),
               source_docs=[chunk["doc_id"]])
        for e in data.get("entities", []) if e.get("name")
    ]
    triples = [
        Triple(
            subject=t["subject"], predicate=t["predicate"], object=t["object"],
            doc_id=chunk["doc_id"], chunk_id=chunk["chunk_id"],
            evidence=t.get("evidence", ""),
        )
        for t in data.get("triples", [])
        if t.get("subject") and t.get("predicate") and t.get("object")
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