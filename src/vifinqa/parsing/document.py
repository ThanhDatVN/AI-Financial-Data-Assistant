from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vifinqa.parsing.metadata import parse_document_metadata
from vifinqa.parsing.models import DocumentMetadata, ParsedTable
from vifinqa.parsing.segment import segment_tables, table_markup_counts
from vifinqa.parsing.table_parser import parse_table


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    metadata: DocumentMetadata
    tables: tuple[ParsedTable, ...]
    table_openings: int
    table_closings: int
    table_matches: int


def parse_document(
    path: Path,
    *,
    data_root: Path,
    first_table_id: int = 1,
    context_lines_before: int = 8,
    max_header_rows: int = 3,
) -> ParsedDocument:
    text = path.read_text(encoding="utf-8", errors="replace")
    metadata = parse_document_metadata(path, data_root=data_root, text=text)
    raw_tables = segment_tables(
        text,
        first_table_id=first_table_id,
        context_lines_before=context_lines_before,
    )
    tables = tuple(parse_table(raw, max_header_rows=max_header_rows) for raw in raw_tables)
    openings, closings, matches = table_markup_counts(text)
    return ParsedDocument(
        metadata=metadata,
        tables=tables,
        table_openings=openings,
        table_closings=closings,
        table_matches=matches,
    )


def statement_paths(data_root: Path) -> list[Path]:
    return sorted((data_root / "financial_statements").glob("*/*/*/*.txt"))
