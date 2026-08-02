from __future__ import annotations

import re
from pathlib import Path

from vifinqa.parsing.models import DocumentMetadata
from vifinqa.parsing.normalize import ascii_compact, normalize_text

_FISCAL_END_RE = re.compile(
    r"(?i)(?:kết\s+thúc|ket\s+thuc)(?:\s+vào)?\s+ngày\s+(\d{1,2}[./-]\d{1,2}[./-]\d{4})"
)


def detect_scope(doc_id: str) -> str:
    compact = ascii_compact(doc_id)
    if "consolidated" in compact or "hopnhat" in compact:
        return "consolidated"
    if "separate" in compact or "rieng" in compact or "congtyme" in compact:
        return "separate"
    if "aggregated" in compact or "tonghop" in compact:
        return "aggregated"
    return "unknown"


def detect_fiscal_year_end(text: str) -> str | None:
    sample = normalize_text(text[:50_000])
    match = _FISCAL_END_RE.search(sample)
    return match.group(1) if match else None


def parse_document_metadata(path: Path, *, data_root: Path, text: str = "") -> DocumentMetadata:
    relative = path.resolve().relative_to(data_root.resolve())
    parts = relative.parts
    if len(parts) < 4:
        raise ValueError(f"Expected TICKER/YEAR/DOCUMENT/file layout: {relative}")
    ticker, raw_year, doc_id = parts[-4], parts[-3], parts[-2]
    if not raw_year.isdigit():
        raise ValueError(f"Invalid report year in path: {relative}")
    return DocumentMetadata(
        ticker=ticker.upper(),
        year=int(raw_year),
        doc_id=doc_id,
        scope=detect_scope(doc_id),
        path=path,
        fiscal_year_end=detect_fiscal_year_end(text) if text else None,
    )
