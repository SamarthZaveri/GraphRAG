"""
Deterministic table parsing for financial filings.

Financial earnings releases are full of whitespace-aligned tables like:

    RECONCILIATION TO ADJUSTED EBITDA (in thousands):
                                        Q2 2025      Q2 2024      H1 2025      H1 2024
    Stock-based compensation           41,859       43,632       81,104       88,316

When this gets flattened into a text chunk for an LLM to read, the model has
to visually re-infer which number belongs to which row/column from spacing
alone -- a small model does this unreliably (this was the root cause behind
a real benchmark failure: both GraphRAG and vector RAG independently missed
Five9's stock-based compensation figure, because the number is easy to lose
track of once flattened, even though it's sitting right there in a table).

This module detects these tables with plain regex/whitespace heuristics (no
LLM call, so it's both more reliable AND faster than asking a model to parse
them), turns every row x column cell into a guaranteed-correct fact, and
strips the table text out of what actually gets sent to the LLM -- which
then only has to handle prose, not numbers-in-a-grid.
"""
from __future__ import annotations
import re
from typing import List, Optional, Tuple

from .models import Entity, Triple

_SPLIT_RE = re.compile(r"\s{2,}")

_PERIOD_PATTERNS = [
    re.compile(r"^Q[1-4]\s+20\d\d$", re.I),
    re.compile(r"^H[12]\s+20\d\d$", re.I),
    re.compile(r"^FY\s?20\d\d$", re.I),
    re.compile(r"^20\d\d$"),
    re.compile(r"^[A-Z][a-z]+\.?\s+\d{1,2},?\s+20\d\d$"),  # "June 30, 2025"
]

_VALUE_RE = re.compile(r"^\(?-?\$?[\d,]+\.?\d*\)?%?$")

# Company-suffix heuristic used to find "whose numbers are these" once per
# document, since our table rows (e.g. "Net revenue") don't repeat the
# company name on every line.
_ENTITY_NAME_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9&\.]*(?:\s+[A-Z][A-Za-z0-9&\.]*){0,3},?\s+"
    r"(?:Inc|Corp|Corporation|LLC|Company|Plc|Co)\.?)"
)


def guess_primary_entity(document_text: str) -> Optional[str]:
    """Best-effort deterministic guess at the document's primary subject
    (e.g. "MaxLinear, Inc.") from the first ~1000 characters, so table rows
    without an explicit company name can still be attributed to one."""
    head = document_text[:1000]
    counts: dict = {}
    for m in _ENTITY_NAME_RE.finditer(head):
        name = m.group(1).rstrip(",")
        counts[name] = counts.get(name, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def _is_period_label(token: str) -> bool:
    return any(p.match(token.strip()) for p in _PERIOD_PATTERNS)


def _is_value(token: str) -> bool:
    return bool(_VALUE_RE.match(token.strip()))


def _split_row(line: str) -> List[str]:
    return [t.strip() for t in _SPLIT_RE.split(line.strip()) if t.strip()]


def detect_and_extract_tables(
    document_text: str, doc_id: str,
) -> Tuple[List[Entity], List[Triple], str]:
    """Scans the FULL document text (before chunking) for whitespace-aligned
    tables. Returns (entities, triples, remaining_text_with_tables_removed).
    Deterministic -- no LLM call."""
    subject = guess_primary_entity(document_text) or ""
    lines = document_text.split("\n")

    entities: List[Entity] = []
    triples: List[Triple] = []
    keep_line = [True] * len(lines)
    table_idx = 0

    i = 0
    while i < len(lines):
        tokens = _split_row(lines[i])
        if len(tokens) >= 2 and sum(_is_period_label(t) for t in tokens) >= max(2, len(tokens) - 1):
            header = tokens
            j = i + 1
            matched_any_row = False
            while j < len(lines):
                row_tokens = _split_row(lines[j])
                if len(row_tokens) < 2:
                    break
                label, values = row_tokens[0], row_tokens[1:]
                if _is_value(label):
                    break  # first token should be a row label, not a number
                value_hits = sum(_is_value(v) for v in values)
                if value_hits == 0:
                    break
                matched_any_row = True
                keep_line[j] = False
                metric_name = f"{subject} {label}".strip()
                if not entities or entities[-1].name != metric_name:
                    entities.append(Entity(name=metric_name, type="Metric",
                                            description=f"Financial metric for {subject}" if subject else label))
                for col, val in zip(header, values):
                    if not _is_value(val):
                        continue
                    period_node = col if _is_period_label(col) else f"{metric_name} ({col})"
                    period_metric = f"{metric_name} {col}"
                    entities.append(Entity(name=period_metric, type="Metric",
                                            description=f"{label} for {subject} in {col}: {val}"))
                    triples.append(Triple(
                        subject=period_metric, predicate="has value", object=val,
                        doc_id=doc_id, chunk_id=f"{doc_id}::table{table_idx}",
                        evidence=f"{label} in {col} was {val}",
                    ))
                    triples.append(Triple(
                        subject=period_metric, predicate="was in period", object=period_node,
                        doc_id=doc_id, chunk_id=f"{doc_id}::table{table_idx}",
                        evidence=f"{label} reported for {col}",
                    ))
                    triples.append(Triple(
                        subject=metric_name, predicate="has period value", object=period_metric,
                        doc_id=doc_id, chunk_id=f"{doc_id}::table{table_idx}",
                        evidence=f"{label} broken out by period",
                    ))
                j += 1
            if matched_any_row:
                keep_line[i] = False
                table_idx += 1
                i = j
                continue
        i += 1

    remaining_text = "\n".join(line for line, keep in zip(lines, keep_line) if keep)
    return entities, triples, remaining_text


def table_chunk_texts(document_text: str, doc_id: str) -> List[dict]:
    """Returns synthetic 'chunk' records (chunk_id, doc_id, text) for each
    detected table, so citations/hybrid-grounding can still point back to
    the original table text even though it bypassed LLM extraction."""
    lines = document_text.split("\n")
    chunks = []
    table_idx = 0
    i = 0
    while i < len(lines):
        tokens = _split_row(lines[i])
        if len(tokens) >= 2 and sum(_is_period_label(t) for t in tokens) >= max(2, len(tokens) - 1):
            start = i
            j = i + 1
            while j < len(lines):
                row_tokens = _split_row(lines[j])
                if len(row_tokens) < 2 or _is_value(row_tokens[0]) or not any(_is_value(v) for v in row_tokens[1:]):
                    break
                j += 1
            if j > i + 1:
                chunks.append({
                    "chunk_id": f"{doc_id}::table{table_idx}", "doc_id": doc_id,
                    "text": "\n".join(lines[start:j]),
                })
                table_idx += 1
            i = j
            continue
        i += 1
    return chunks