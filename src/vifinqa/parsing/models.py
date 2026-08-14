from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DocumentMetadata:
    ticker: str
    year: int
    doc_id: str
    scope: str
    path: Path
    fiscal_year_end: str | None = None


@dataclass(frozen=True, slots=True)
class RawTable:
    table_id: int
    page_no: int | None
    line_no: int
    char_offset: int
    html: str
    context_before: tuple[str, ...]
    section_title: str | None
    unit_declaration: str | None = None


@dataclass(frozen=True, slots=True)
class ParsedTable:
    raw: RawTable
    matrix: tuple[tuple[str, ...], ...]
    header_rows: int
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    unit: str

    @property
    def n_rows(self) -> int:
        return len(self.matrix)

    @property
    def n_cols(self) -> int:
        return len(self.headers)
