from __future__ import annotations

import re
from collections.abc import Iterable

from vifinqa.parsing.normalize import ascii_compact, ascii_words, normalize_text

TABLE_UNIT_INFERENCE_VERSION = 3

_UNIT_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("MILLION_USD", ("trieuusd", "millionusd", "trieudolamy")),
    ("BILLION_VND", ("tydong", "tyvnd")),
    ("MILLION_VND", ("trieudong", "trieuvnd")),
    ("THOUSAND_VND", ("nghindong", "ngandong", "nghinvnd")),
    ("PERCENT", ("phantram", "tyle", "tysuat", "%")),
    ("USD", ("usd", "usdollar", "dolamy")),
    ("VND", ("vnd", "vietnamdong", "donvidong", "dongcophieu", "dongcophan")),
    ("SHARES", ("cophieu", "cophan")),
)

SOURCE_MULTIPLIER: dict[str, float] = {
    "VND": 1.0,
    "THOUSAND_VND": 1e3,
    "MILLION_VND": 1e6,
    "BILLION_VND": 1e9,
    "USD": 1.0,
    "MILLION_USD": 1e6,
    "SHARES": 1.0,
    "PERCENT": 1.0,
    "UNKNOWN": 1.0,
}

_UNIT_DECLARATION_RE = re.compile(r"^(?:don vi(?: tinh)?|dvt|unit|currency)\b")
_SHARE_VALUE_MARKERS = (
    "giatri",
    "menhgia",
    "giacophieu",
    "voncophan",
    "dautu",
    "loinhuan",
)


def is_unit_declaration(line: str) -> bool:
    """Whether a line explicitly declares a default unit rather than merely mentioning one."""
    return _UNIT_DECLARATION_RE.match(ascii_words(normalize_text(line))) is not None


def detect_source_unit(lines: Iterable[str]) -> str:
    candidates = [normalize_text(line) for line in lines if normalize_text(line)]
    # The nearest context/header line is normally the most specific source.
    for line in reversed(candidates):
        compact = ascii_compact(line)
        for unit, patterns in _UNIT_PATTERNS:
            # A share noun is not itself a unit when the label names a monetary value.
            # Mixed capital tables commonly pair "Số cổ phần" with "Giá trị cổ phần";
            # only the former is a quantity of shares. An explicit VND marker, if present,
            # has already matched above because monetary units precede SHARES.
            if unit == "SHARES" and any(marker in compact for marker in _SHARE_VALUE_MARKERS):
                continue
            if any(pattern in compact or pattern in line.lower() for pattern in patterns):
                return unit
        if compact == "dong":
            return "VND"
    return "UNKNOWN"


def detect_table_unit(
    context_lines: Iterable[str],
    headers: Iterable[str],
    *,
    inherited_declaration: str | None = None,
) -> str:
    """Infer a table default without leaking one mixed column's unit across every cell.

    An explicit nearby declaration is authoritative. Otherwise a unit present consistently in
    the headers is a usable default; mixed headers are deliberately UNKNOWN and resolved per cell.
    Narrative mentions such as "phát hành cổ phiếu ... đồng" are not unit declarations.
    """
    context = tuple(context_lines)
    for line in reversed(context):
        if is_unit_declaration(line):
            declared = detect_source_unit((line,))
            if declared != "UNKNOWN":
                return declared
    if inherited_declaration is not None:
        inherited = detect_source_unit((inherited_declaration,))
        if inherited != "UNKNOWN":
            return inherited
    header_units = {
        unit for header in headers if (unit := detect_source_unit((header,))) != "UNKNOWN"
    }
    return next(iter(header_units)) if len(header_units) == 1 else "UNKNOWN"


def resolve_cell_unit(table_unit: str, *, row_label: str, column_label: str) -> str:
    """Prefer a cell's column/row currency marker over the table-level default."""
    column_unit = detect_source_unit((column_label,))
    if column_unit != "UNKNOWN":
        return column_unit
    row_unit = detect_source_unit((row_label,))
    if row_unit != "UNKNOWN":
        return row_unit
    return table_unit


def to_base_unit(value: float, unit: str) -> float:
    return value * SOURCE_MULTIPLIER[unit]
