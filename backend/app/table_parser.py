"""
Deterministic table parsing for financial filings.

Tables are where filings keep their numbers, and an LLM reading a flattened grid drops rows
(a real FinQA failure: extraction kept 13 facts from a table chunk but not the "total" row the
question asked about). Every table is therefore parsed without an LLM into one fact per
row x column cell, cited to a table chunk, and removed from the prose sent to LLM extraction.

Three layouts are recognized and normalized to one row model:

  1. Labeled rows, as written by prepare_finqa.render_document and html_text.html_to_text:
         TABLE HEADERS: ($ in millions) | 3Q26 | 3Q25
         TABLE ROW Operating revenue | 3Q26: 20,186 ; 3Q25: 16,673
  2. TABLE SECTION lines inside such tables, naming a block of rows ("Operating expense").
  3. Whitespace-aligned text tables whose header row is mostly period labels:
                                   Q2 2025      Q2 2024
         Stock-based compensation   41,859       43,632
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
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
_VALUE_RE = re.compile(r"^\(?\s*[$€£]?\s*\(?\s*[-–—]?\s*\d[\d,]*(?:\.\d+)?\s*\)?\s*(?:%|x|pts?|bps)?\s*\)?$", re.I)
_UNIT_RE = re.compile(r"\bin (thousands|millions|billions)\b", re.I)
_UNSCALED_LABEL = re.compile(r"per share|per gallon|\(cents\)|margin|ratio|percent|%|rate\b|yield|per available|days|employees", re.I)
_GENERIC_LABEL = re.compile(r"^(total|subtotal|other|net|change|% change)\b", re.I)
_ENTITY_NAME_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9&\.]*(?:\s+[A-Z][A-Za-z0-9&\.]*){0,3},?\s+"
    r"(?:Inc|Corp|Corporation|LLC|Company|Plc|Co)\.?)"
)
MAX_ROWS_PER_CHUNK = 40


@dataclass
class Table:
    caption: str = ""
    headers: List[str] = field(default_factory=list)
    rows: List[Tuple[str, List[Tuple[str, str]]]] = field(default_factory=list)  # (label, [(column, value)])
    lines: List[str] = field(default_factory=list)
    data_headers: set = field(default_factory=set)

    @property
    def unit(self) -> str:
        match = _UNIT_RE.search(" ".join([self.caption] + self.headers))
        return match.group(1).lower() if match else ""


def guess_primary_entity(document_text: str) -> Optional[str]:
    """Whose numbers these are: an explicit COMPANY: line, a FinQA SOURCE ticker, or the most
    frequent "<Name> Inc."-style name in the opening text."""
    head = document_text[:1500]
    explicit = re.search(r"^COMPANY:\s*(.+)$", head, re.M)
    if explicit:
        return re.sub(r"\s*\([A-Z.]+\)\s*$", "", explicit.group(1).strip())
    source = re.match(r"\s*SOURCE:\s*([A-Z.]{1,6})/\d{4}/", head)
    if source:
        return source.group(1)
    counts: dict = {}
    for m in _ENTITY_NAME_RE.finditer(head[:1000]):
        name = m.group(1).rstrip(",")
        counts[name] = counts.get(name, 0) + 1
    return max(counts, key=counts.get) if counts else None


def _is_period_label(token: str) -> bool:
    return any(p.match(token.strip()) for p in _PERIOD_PATTERNS)


def _is_value(token: str) -> bool:
    return bool(_VALUE_RE.match(token.strip()))


def _clean_value(value: str) -> str:
    # FinQA writes "$ 1,234" and "( 56 )"; keep the figure exactly, only drop inner spaces.
    return re.sub(r"\s+", "", value.strip())


def _split_row(line: str) -> List[str]:
    return [t.strip() for t in _SPLIT_RE.split(line.strip()) if t.strip()]


def _parse_labeled_row(body: str) -> Tuple[str, List[Tuple[str, str]]]:
    label, _, cells = body.partition(" | ")
    values = []
    for cell in cells.split(" ; ") if cells else []:
        column, sep, value = cell.rpartition(": ")
        if sep and value.strip():
            values.append((column.strip(), value.strip()))
    return label.strip(), values


def _labeled_tables(lines: List[str], keep: List[bool]) -> List[Table]:
    tables, current, section, last_prose = [], None, "", ""
    for i, raw in enumerate(lines):
        line = raw.strip()
        is_table_line = line.startswith(("TABLE HEADERS:", "TABLE ROW ", "TABLE SECTION "))
        if not is_table_line:
            if line and current is not None:
                current = None  # prose ends the table; blank lines do not
            if line:
                last_prose = line
            continue
        keep[i] = False
        if current is None or line.startswith("TABLE HEADERS:") and current.rows:
            current = Table(caption=last_prose if len(last_prose) <= 200 else "")
            tables.append(current)
            section = ""
        current.lines.append(line)
        if line.startswith("TABLE HEADERS:"):
            items = [h.strip() for h in line[len("TABLE HEADERS:"):].split(" | ")]
            if len(items) > 1 and all(_is_value(h) for h in items[1:]):
                # FinQA tables without a header row: the first row was rendered as headers, and
                # later rows name their columns by those values.
                current.rows.append((items[0], [("value", h) for h in items[1:]]))
                current.data_headers = set(items[1:])
                current.headers = []
            else:
                current.headers = items
        elif line.startswith("TABLE SECTION "):
            section = line[len("TABLE SECTION "):].strip().rstrip(":")
        else:
            label, values = _parse_labeled_row(line[len("TABLE ROW "):])
            values = [("value" if column in current.data_headers else column, value) for column, value in values]
            if _GENERIC_LABEL.match(label) and section:
                label = f"{section} - {label}"
            if values:
                current.rows.append((label, values))
    return [t for t in tables if t.rows]


def _whitespace_tables(lines: List[str], keep: List[bool]) -> List[Table]:
    tables, i = [], 0
    while i < len(lines):
        tokens = _split_row(lines[i]) if keep[i] else []
        if len(tokens) >= 2 and sum(_is_period_label(t) for t in tokens) >= max(2, len(tokens) - 1):
            table = Table(headers=tokens, lines=[lines[i].strip()])
            j = i + 1
            while j < len(lines) and keep[j]:
                row = _split_row(lines[j])
                if len(row) < 2 or _is_value(row[0]) or not any(_is_value(v) for v in row[1:]):
                    break
                table.rows.append((row[0], [(col, val) for col, val in zip(tokens, row[1:]) if _is_value(val)]))
                table.lines.append(lines[j].strip())
                j += 1
            if table.rows:
                for k in range(i, j):
                    keep[k] = False
                tables.append(table)
                i = j
                continue
        i += 1
    return tables


def parse_tables(document_text: str) -> Tuple[List[Table], str]:
    """All tables in a document, plus the prose with table lines removed."""
    lines = document_text.split("\n")
    keep = [True] * len(lines)
    tables = _labeled_tables(lines, keep) + _whitespace_tables(lines, keep)
    remaining = "\n".join(line for line, k in zip(lines, keep) if k)
    return tables, re.sub(r"\n{3,}", "\n\n", remaining)


def _chunk_ranges(table: Table):
    for start in range(0, len(table.rows), MAX_ROWS_PER_CHUNK):
        yield start, min(start + MAX_ROWS_PER_CHUNK, len(table.rows))


def detect_and_extract_tables(document_text: str, doc_id: str) -> Tuple[List[Entity], List[Triple], str]:
    """Deterministic facts from every table: (entities, triples, prose_without_tables).
    Each cell becomes a literal-valued triple, which the graph stores on the metric node
    rather than as a shared value node."""
    subject = guess_primary_entity(document_text) or ""
    tables, remaining = parse_tables(document_text)
    entities: List[Entity] = []
    triples: List[Triple] = []
    if subject:
        entities.append(Entity(name=subject, type="Company", description="Reporting company"))
    seen = set()
    chunk_index = 0
    for table in tables:
        unit = f" (in {table.unit})" if table.unit else ""
        scope = table.caption[:80]
        for start, end in _chunk_ranges(table):
            chunk_id = f"{doc_id}::table{chunk_index}"
            chunk_index += 1
            for label, values in table.rows[start:end]:
                metric = f"{subject} {label}".strip() if subject else label
                if metric not in seen:
                    seen.add(metric)
                    entities.append(Entity(name=metric, type="Metric",
                                           description=(f"{label}{unit}" + (f"; table: {scope}" if scope else ""))))
                    if subject:
                        triples.append(Triple(subject=subject, predicate="reports", object=metric,
                                              doc_id=doc_id, chunk_id=chunk_id, evidence=f"{subject} reports {label}"))
                # Table scale captions ("$ in millions except per share") do not apply to rates.
                scaled = unit and not _UNSCALED_LABEL.search(label)
                for column, value in values:
                    value = _clean_value(value)
                    note = unit if scaled and "%" not in value and "%" not in column else ""
                    triples.append(Triple(subject=metric, predicate=column, object=value, doc_id=doc_id,
                                          chunk_id=chunk_id, evidence=f"{label}, {column}: {value}{note}"))
    return entities, triples, remaining


def table_chunk_texts(document_text: str, doc_id: str) -> List[dict]:
    """Citable chunk records for each table (split into row windows that repeat the header line),
    in the same order and with the same ids as detect_and_extract_tables' citations."""
    tables, _ = parse_tables(document_text)
    chunks, chunk_index = [], 0
    for table in tables:
        # Rows are re-rendered from the parsed model so each chunk holds exactly the cells
        # (with section-qualified labels) whose facts cite it.
        header = ["TABLE HEADERS: " + " | ".join(table.headers)] if table.headers else []
        for start, end in _chunk_ranges(table):
            rows = ["TABLE ROW " + label + " | " + " ; ".join(f"{c}: {_clean_value(v)}" for c, v in values)
                    for label, values in table.rows[start:end]]
            text = "\n".join(([table.caption] if table.caption else []) + header + rows)
            chunks.append({"chunk_id": f"{doc_id}::table{chunk_index}", "doc_id": doc_id, "text": text})
            chunk_index += 1
    return chunks
