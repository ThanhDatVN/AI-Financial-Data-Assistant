from __future__ import annotations

from collections.abc import Iterable

from lxml import html as lxml_html

from vifinqa.parsing.models import ParsedTable, RawTable
from vifinqa.parsing.normalize import normalize_text
from vifinqa.parsing.numbers import is_numeric_cell
from vifinqa.parsing.units import detect_source_unit


def _positive_int(raw: object, default: int = 1) -> int:
    try:
        value = int(str(raw))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def html_to_matrix(html: str) -> tuple[tuple[str, ...], ...]:
    try:
        root = lxml_html.fromstring(html)
    except (TypeError, ValueError, lxml_html.ParserError):
        return ()
    tables = [root] if root.tag.lower() == "table" else root.xpath(".//table")
    if not tables:
        return ()
    table = tables[0]

    # col -> (remaining future rows, value)
    row_spans: dict[int, tuple[int, str]] = {}
    rows: list[dict[int, str]] = []
    for tr in table.xpath(".//tr"):
        current: dict[int, str] = {}
        next_spans: dict[int, tuple[int, str]] = {}
        for col, (remaining, value) in row_spans.items():
            current[col] = value
            if remaining > 1:
                next_spans[col] = (remaining - 1, value)

        col = 0
        cells = tr.xpath("./th|./td")
        if not cells:
            cells = tr.xpath(".//th|.//td")
        for cell in cells:
            while col in current:
                col += 1
            value = normalize_text(" ".join(cell.itertext()))
            colspan = _positive_int(cell.get("colspan"))
            rowspan = _positive_int(cell.get("rowspan"))
            for offset in range(colspan):
                target = col + offset
                current[target] = value
                if rowspan > 1:
                    next_spans[target] = (rowspan - 1, value)
            col += colspan
        row_spans = next_spans
        if current:
            rows.append(current)

    if not rows:
        return ()
    width = max(max(row) for row in rows) + 1
    return tuple(tuple(row.get(col, "") for col in range(width)) for row in rows)


def _header_score(row: Iterable[str]) -> float:
    cells = [cell for cell in row if cell]
    if not cells:
        return 0.0
    numeric = sum(is_numeric_cell(cell) for cell in cells)
    return 1.0 - numeric / len(cells)


def infer_header_rows(matrix: tuple[tuple[str, ...], ...], *, max_header_rows: int = 3) -> int:
    if not matrix:
        return 0
    header_rows = 1
    for index, row in enumerate(matrix[1:max_header_rows], start=1):
        if _header_score(row) >= 0.75:
            header_rows = index + 1
        else:
            break
    return min(header_rows, len(matrix))


def _deduplicate_headers(headers: list[str]) -> tuple[str, ...]:
    seen: dict[str, int] = {}
    output: list[str] = []
    for index, raw in enumerate(headers, start=1):
        base = raw or f"column_{index}"
        count = seen.get(base, 0) + 1
        seen[base] = count
        output.append(base if count == 1 else f"{base}__{count}")
    return tuple(output)


def combine_headers(matrix: tuple[tuple[str, ...], ...], header_rows: int) -> tuple[str, ...]:
    if not matrix:
        return ()
    width = len(matrix[0])
    headers: list[str] = []
    for col in range(width):
        parts: list[str] = []
        for row in matrix[:header_rows]:
            value = row[col]
            if value and (not parts or value != parts[-1]):
                parts.append(value)
        headers.append(" | ".join(parts))
    return _deduplicate_headers(headers)


def parse_table(raw: RawTable, *, max_header_rows: int = 3) -> ParsedTable:
    matrix = html_to_matrix(raw.html)
    header_rows = infer_header_rows(matrix, max_header_rows=max_header_rows)
    headers = combine_headers(matrix, header_rows)
    rows = matrix[header_rows:]
    unit_context = (*raw.context_before, *headers)
    return ParsedTable(
        raw=raw,
        matrix=matrix,
        header_rows=header_rows,
        headers=headers,
        rows=rows,
        unit=detect_source_unit(unit_context),
    )
