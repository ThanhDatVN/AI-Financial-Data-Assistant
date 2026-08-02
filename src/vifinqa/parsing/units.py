from __future__ import annotations

from collections.abc import Iterable

from vifinqa.parsing.normalize import ascii_compact, normalize_text

_UNIT_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("MILLION_USD", ("trieuusd", "millionusd", "trieudolamy")),
    ("BILLION_VND", ("tydong", "tyvnd")),
    ("MILLION_VND", ("trieudong", "trieuvnd")),
    ("THOUSAND_VND", ("nghindong", "ngandong", "nghinvnd")),
    ("PERCENT", ("phantram", "%")),
    ("SHARES", ("cophieu",)),
    ("USD", ("usd", "usdollar", "dolamy")),
    ("VND", ("vnd", "vietnamdong", "donvidong")),
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


def detect_source_unit(lines: Iterable[str]) -> str:
    candidates = [normalize_text(line) for line in lines if normalize_text(line)]
    # The nearest context/header line is normally the most specific source.
    for line in reversed(candidates):
        compact = ascii_compact(line)
        for unit, patterns in _UNIT_PATTERNS:
            if any(pattern in compact or pattern in line.lower() for pattern in patterns):
                return unit
        if compact == "dong":
            return "VND"
    return "UNKNOWN"


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
