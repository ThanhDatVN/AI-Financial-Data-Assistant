from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from vifinqa.indexing.manifest import ManifestRecord, iter_manifest
from vifinqa.parsing.models import ParsedTable
from vifinqa.parsing.numbers import parse_financial_number
from vifinqa.parsing.segment import segment_tables
from vifinqa.parsing.table_parser import parse_table
from vifinqa.parsing.units import resolve_cell_unit, to_base_unit


def parsed_table_to_long_frame(record: ManifestRecord, table: ParsedTable) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row_index, row in enumerate(table.rows):
        row_label = row[0] if row else ""
        for column_index, raw_value in enumerate(row):
            parsed = parse_financial_number(raw_value)
            numeric_value = parsed.value
            column_label = (
                table.headers[column_index]
                if column_index < len(table.headers)
                else f"column_{column_index + 1}"
            )
            source_unit = resolve_cell_unit(
                table.unit,
                row_label=row_label,
                column_label=column_label,
            )
            base_value = (
                to_base_unit(numeric_value, source_unit) if numeric_value is not None else None
            )
            rows.append(
                {
                    "table_ref": record.table_ref,
                    "doc_id": record.doc_id,
                    "ticker": record.ticker,
                    "report_year": record.report_year,
                    "scope": record.scope,
                    "source_unit": source_unit,
                    "row_index": row_index,
                    "column_index": column_index,
                    "row_label": row_label,
                    "column_label": column_label,
                    "raw_value": raw_value,
                    "numeric_value": numeric_value,
                    "base_value": base_value,
                    "is_dash": parsed.is_dash,
                    "is_missing": parsed.is_missing,
                }
            )
    return pd.DataFrame(rows)


class TableStore:
    def __init__(self, data_root: Path, records: list[ManifestRecord]) -> None:
        self.data_root = data_root
        self.records = {record.table_ref: record for record in records}
        if len(self.records) != len(records):
            raise ValueError("Duplicate table_ref values in manifest")

    @classmethod
    def from_manifest(cls, data_root: Path, manifest_path: Path) -> TableStore:
        return cls(data_root, list(iter_manifest(manifest_path)))

    @classmethod
    def from_parquet(cls, data_root: Path, manifest_path: Path, table_refs: set[str]) -> TableStore:
        table = pq.read_table(manifest_path, filters=[("table_ref", "in", list(table_refs))])
        records = [ManifestRecord.from_dict(row) for row in table.to_pylist()]
        missing = table_refs - {record.table_ref for record in records}
        if missing:
            raise KeyError(f"Unknown table_ref values: {sorted(missing)}")
        return cls(data_root, records)

    def load(self, table_ref: str) -> tuple[ManifestRecord, ParsedTable]:
        try:
            record = self.records[table_ref]
        except KeyError as exc:
            raise KeyError(f"Unknown table_ref: {table_ref}") from exc
        path = self.data_root / Path(record.source_path)
        text = path.read_text(encoding="utf-8", errors="replace")
        raw_tables = segment_tables(text, first_table_id=1)
        offset = record.table_id - 1
        if offset < 0 or offset >= len(raw_tables):
            raise ValueError(f"table_id {record.table_id} is outside {path}")
        raw = raw_tables[offset]
        digest = hashlib.sha256(raw.html.encode("utf-8")).hexdigest()
        if digest != record.html_sha256:
            raise ValueError(f"Source drift for {table_ref}: HTML hash differs from manifest")
        return record, parse_table(raw)

    def export_csv(self, table_ref: str, output_path: Path) -> Path:
        record, table = self.load(table_ref)
        frame = parsed_table_to_long_frame(record, table)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False, encoding="utf-8")
        return output_path
