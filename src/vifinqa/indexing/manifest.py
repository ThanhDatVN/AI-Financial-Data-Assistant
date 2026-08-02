from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vifinqa.parsing.document import statement_paths
from vifinqa.parsing.metadata import parse_document_metadata
from vifinqa.parsing.models import ParsedTable
from vifinqa.parsing.normalize import ascii_words, normalize_text
from vifinqa.parsing.numbers import is_numeric_cell
from vifinqa.parsing.segment import segment_tables
from vifinqa.parsing.table_parser import parse_table


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    table_ref: str
    doc_id: str
    ticker: str
    report_year: int
    scope: str
    table_id: int
    page_no: int | None
    line_no: int
    char_offset: int
    section_title: str | None
    unit: str
    header_rows: int
    n_rows: int
    n_cols: int
    headers: tuple[str, ...]
    row_labels: tuple[str, ...]
    retrieval_text: str
    source_path: str
    html_sha256: str
    identity_status: str = "raw_ordinal_1_based_unverified_on_dashboard"

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ManifestRecord:
        raw["headers"] = tuple(raw["headers"])
        raw["row_labels"] = tuple(raw["row_labels"])
        return cls(**raw)


def _unique_non_numeric(values: Iterable[str], *, limit: int = 80) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = normalize_text(raw)
        key = value.casefold()
        if not value or is_numeric_cell(value) or key in seen:
            continue
        output.append(value)
        seen.add(key)
        if len(output) >= limit:
            break
    return tuple(output)


def table_retrieval_text(table: ParsedTable, *, ticker: str, year: int, scope: str) -> str:
    context = _unique_non_numeric(table.raw.context_before[-4:], limit=4)
    row_labels = _unique_non_numeric((cell for row in table.rows for cell in row), limit=80)
    parts = [
        f"ma co phieu: {ticker}",
        f"nam bao cao: {year}",
        f"pham vi: {scope}",
    ]
    if table.raw.section_title:
        parts.append(f"muc: {table.raw.section_title}")
    if context:
        parts.append("ngu canh: " + " | ".join(context))
    if table.headers:
        parts.append("cot: " + " | ".join(table.headers))
    if row_labels:
        parts.append("dong: " + " | ".join(row_labels))
    original = "\n".join(parts)
    return original + "\nkhong dau: " + ascii_words(original)


def record_from_table(
    table: ParsedTable,
    *,
    doc_id: str,
    ticker: str,
    year: int,
    scope: str,
    source_path: Path,
    table_ref_format: str = "{doc_id}|table_{table_id}",
) -> ManifestRecord:
    labels = _unique_non_numeric((cell for row in table.rows for cell in row), limit=80)
    return ManifestRecord(
        table_ref=table_ref_format.format(doc_id=doc_id, table_id=table.raw.table_id),
        doc_id=doc_id,
        ticker=ticker,
        report_year=year,
        scope=scope,
        table_id=table.raw.table_id,
        page_no=table.raw.page_no,
        line_no=table.raw.line_no,
        char_offset=table.raw.char_offset,
        section_title=table.raw.section_title,
        unit=table.unit,
        header_rows=table.header_rows,
        n_rows=table.n_rows,
        n_cols=table.n_cols,
        headers=table.headers,
        row_labels=labels,
        retrieval_text=table_retrieval_text(table, ticker=ticker, year=year, scope=scope),
        source_path=source_path.as_posix(),
        html_sha256=hashlib.sha256(table.raw.html.encode("utf-8")).hexdigest(),
    )


def iter_records(
    data_root: Path,
    *,
    first_table_id: int = 1,
    table_ref_format: str = "{doc_id}|table_{table_id}",
    limit_documents: int | None = None,
    limit_tables: int | None = None,
) -> Iterator[ManifestRecord]:
    emitted = 0
    paths = statement_paths(data_root)
    if limit_documents is not None:
        paths = paths[:limit_documents]
    for path in paths:
        for record in records_for_document(
            path,
            data_root=data_root,
            first_table_id=first_table_id,
            table_ref_format=table_ref_format,
        ):
            yield record
            emitted += 1
            if limit_tables is not None and emitted >= limit_tables:
                return


def records_for_document(
    path: Path,
    *,
    data_root: Path,
    first_table_id: int = 1,
    table_ref_format: str = "{doc_id}|table_{table_id}",
) -> list[ManifestRecord]:
    text = path.read_text(encoding="utf-8", errors="replace")
    metadata = parse_document_metadata(path, data_root=data_root, text=text)
    return [
        record_from_table(
            parse_table(raw_table),
            doc_id=metadata.doc_id,
            ticker=metadata.ticker,
            year=metadata.year,
            scope=metadata.scope,
            source_path=path.relative_to(data_root),
            table_ref_format=table_ref_format,
        )
        for raw_table in segment_tables(text, first_table_id=first_table_id)
    ]


def iter_manifest(path: Path) -> Iterator[ManifestRecord]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield ManifestRecord.from_dict(json.loads(line))
