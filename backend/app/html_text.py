"""Convert filing HTML (e.g. SEC 8-K EX-99.1 earnings releases) to text with tables as labeled rows.

Tables are rendered in the same line format FinQA documents use, so table_parser.py parses
both deterministically:

    TABLE HEADERS: Three Months Ended September 30, 2026 | Three Months Ended September 30, 2025
    TABLE ROW Operating revenue | Three Months Ended September 30, 2026: $16,672 ; ...: $15,677

EDGAR tables split one figure across cells ("$" | "1,234" | ")" ) and use colspans for multi-row
headers, so each value is placed on an expanded column grid and named by the header text that
spans its grid column, rather than by its position among non-empty cells.
"""
from __future__ import annotations
import re
from html.parser import HTMLParser

BLOCK_TAGS = {"p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "center"}
SKIP_TAGS = {"script", "style", "head", "title", "noscript"}
VALUE_RE = re.compile(r"^[\s($€£]*[-–—]?\s*\d[\d,]*(?:\.\d+)?\s*\)?\s*(?:%|x|pts?|bps)?\s*\)?$", re.I)
YEAR_RE = re.compile(r"^(19|20)\d{2}$")
DASHES = {"-", "–", "—", "——", "n/m", "nm", "*"}
GENERIC_LABELS = re.compile(r"^(total|subtotal|other|net|adjusted|as reported|change|% change)\b", re.I)


def _clean(text):
    return re.sub(r"\s+", " ", text.replace(" ", " ")).strip()


def is_value(text):
    return bool(VALUE_RE.match(text.strip()))


class _Converter(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.buffer, self.skip = [], [], 0
        self.tables, self.row, self.cell, self.colspan = [], None, None, 1

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self.skip += 1
        elif tag == "table":
            self._flush()
            self.tables.append([])
        elif tag == "tr" and self.tables:
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = []
            try:
                self.colspan = max(1, min(int(dict(attrs).get("colspan") or 1), 50))
            except ValueError:
                self.colspan = 1
        elif tag in BLOCK_TAGS:
            if self.cell is not None:
                self.cell.append(" ")
            else:
                self._flush()

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS:
            self.skip = max(0, self.skip - 1)
        elif tag in ("td", "th") and self.cell is not None and self.row is not None:
            self.row.append((_clean("".join(self.cell)), self.colspan))
            self.cell = None
        elif tag == "tr" and self.row is not None and self.tables:
            if any(text for text, _ in self.row):
                self.tables[-1].append(self.row)
            self.row = None
        elif tag == "table" and self.tables:
            rendered = render_table(self.tables.pop())
            if rendered:
                self.out.append(rendered)
        elif tag in BLOCK_TAGS and self.cell is None:
            self._flush()

    def handle_data(self, data):
        if self.skip:
            return
        if self.cell is not None:
            self.cell.append(data)
        elif not self.tables:
            self.buffer.append(data)

    def _flush(self):
        text = _clean("".join(self.buffer))
        if text:
            self.out.append(text)
        self.buffer = []


def _grid(row):
    """Expand colspans: [(column_index, text)], one entry per cell at its first grid column."""
    cells, column = [], 0
    for text, span in row:
        cells.append((column, span, text))
        column += span
    return cells


def _merge_fragments(cells):
    """Join '$' / '(' prefixes and ')' / '%' suffixes split into their own cells."""
    merged = []
    pending = ""
    for column, span, text in cells:
        if not text:
            continue
        if text in ("$", "(", "($", "$(", "€", "£"):
            pending += text
            continue
        if merged and re.fullmatch(r"\)?\s*%?\)?", text) and text:
            c, s, t = merged[-1]
            merged[-1] = (c, s, t + text)
            continue
        merged.append((column, span, (pending + text) if pending else text))
        pending = ""
    return merged


def render_table(rows):
    if not rows:
        return ""
    grids = [_merge_fragments(_grid(row)) for row in rows]

    def data_row(g):
        values = [t for _, _, t in g[1:] if is_value(t)]
        # A row of bare years ("2026 | 2025") is a column header, not figures.
        return bool(values) and not all(YEAR_RE.match(t) for t in values)
    is_data = [data_row(g) for g in grids]
    if not any(is_data):
        # Layout table (or a text box): keep its text as prose.
        return "\n\n".join(" ".join(t for _, _, t in g) for g in grids if g)
    first_data = is_data.index(True)
    width = max((c + s for g in grids for c, s, _ in g), default=0)
    headers = [""] * width
    # Header rows (including year-like numeric cells such as "2026") name every grid column they span.
    for g in grids[:first_data]:
        for c, s, t in g:
            for k in range(c, min(c + s, width)):
                headers[k] = f"{headers[k]} {t}".strip()

    def header_for(c, s):
        for k in range(c, min(c + s, width)):
            if headers[k]:
                return headers[k]
        # A value can sit one grid column beside its header span (after a split-off "$" cell).
        for k in (c - 1, c + 1):
            if 0 <= k < width and headers[k]:
                return headers[k]
        return ""

    lines = []
    unique_headers = [h for i, h in enumerate(headers) if h and h not in headers[:i]]
    if unique_headers:
        lines.append("TABLE HEADERS: " + " | ".join(unique_headers))
    section = ""
    for g, data in zip(grids[first_data:], is_data[first_data:]):
        if not g:
            continue
        label_cells = [(c, s, t) for c, s, t in g if not is_value(t) and t.lower() not in DASHES]
        values = [(c, s, t) for c, s, t in g if is_value(t) or t.lower() in DASHES]
        label = label_cells[0][2] if label_cells and label_cells[0][0] <= (values[0][0] if values else width) else ""
        if not data or not values:
            section = label.rstrip(":") if label else section
            if label:
                lines.append("TABLE SECTION " + label)
            continue
        if GENERIC_LABELS.match(label) and section:
            label = f"{section} - {label}"
        named = []
        for c, s, t in values:
            if t.lower() in DASHES:
                continue
            named.append(f"{header_for(c, s) or 'value'}: {t}")
        if named:
            lines.append("TABLE ROW " + (label or "(unlabeled)") + " | " + " ; ".join(named))
    return "\n".join(lines)


def html_to_text(html):
    converter = _Converter()
    converter.feed(html)
    converter.close()
    converter._flush()
    text = "\n\n".join(converter.out)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
