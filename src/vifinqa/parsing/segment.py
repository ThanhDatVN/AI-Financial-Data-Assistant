from __future__ import annotations

import bisect
import re

from vifinqa.parsing.models import RawTable
from vifinqa.parsing.normalize import normalize_text

_PAGE_RE = re.compile(r"(?m)^=====\s*PAGE\s+(\d+)\s*=====\s*$", re.IGNORECASE)
_TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table\s*>", re.IGNORECASE | re.DOTALL)
_HTML_ONLY_RE = re.compile(r"^<[^>]+>$")
_SECTION_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)*\.?\s+\S")


def _line_starts(text: str) -> list[int]:
    starts = [0]
    starts.extend(match.end() for match in re.finditer(r"\n", text))
    return starts


def _context_before(lines: list[str], line_index: int, *, max_lines: int) -> tuple[str, ...]:
    context: list[str] = []
    for index in range(line_index - 1, -1, -1):
        raw = lines[index]
        line = normalize_text(raw)
        # A preceding table is a hard context boundary. Carrying its HTML into the
        # next table can leak old headers/units (for example "%" or "VND").
        if re.search(r"</table\s*>", line, re.IGNORECASE):
            break
        if not line or _PAGE_RE.fullmatch(line):
            continue
        context.append(line)
        if len(context) >= max_lines:
            break
    return tuple(reversed(context))


def _section_title(context: tuple[str, ...]) -> str | None:
    for line in reversed(context):
        if _HTML_ONLY_RE.fullmatch(line):
            continue
        if _SECTION_NUMBER_RE.match(line):
            return line
    return next((line for line in reversed(context) if not line.startswith("<")), None)


def segment_tables(
    text: str,
    *,
    first_table_id: int = 1,
    context_lines_before: int = 8,
) -> tuple[RawTable, ...]:
    line_starts = _line_starts(text)
    lines = text.splitlines()
    page_matches = list(_PAGE_RE.finditer(text))
    page_offsets = [match.start() for match in page_matches]
    tables: list[RawTable] = []
    previous_table_end = 0
    current_numbered_section: str | None = None
    for ordinal, match in enumerate(_TABLE_RE.finditer(text), start=first_table_id):
        between_tables = text[previous_table_end : match.start()]
        for raw_line in between_tables.splitlines():
            line = normalize_text(raw_line)
            if _SECTION_NUMBER_RE.match(line):
                current_numbered_section = line
        page_index = bisect.bisect_right(page_offsets, match.start()) - 1
        page_no = int(page_matches[page_index].group(1)) if page_index >= 0 else None
        line_index = bisect.bisect_right(line_starts, match.start()) - 1
        context = _context_before(lines, line_index, max_lines=context_lines_before)
        local_section = _section_title(context)
        tables.append(
            RawTable(
                table_id=ordinal,
                page_no=page_no,
                line_no=line_index + 1,
                char_offset=match.start(),
                html=match.group(0),
                context_before=context,
                section_title=current_numbered_section or local_section,
            )
        )
        previous_table_end = match.end()
    return tuple(tables)


def table_markup_counts(text: str) -> tuple[int, int, int]:
    openings = len(re.findall(r"<table\b", text, re.IGNORECASE))
    closings = len(re.findall(r"</table\s*>", text, re.IGNORECASE))
    matched = len(_TABLE_RE.findall(text))
    return openings, closings, matched
