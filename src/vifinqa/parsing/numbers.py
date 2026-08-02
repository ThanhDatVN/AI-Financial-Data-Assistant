from __future__ import annotations

import math
import re
from dataclasses import dataclass

from vifinqa.parsing.normalize import normalize_text

_DASHES = {"-", "–", "—", "−"}
_CURRENCY_RE = re.compile(r"(?i)\s*(vnd|vnđ|đồng|dong)\s*$")
_NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:[.,]\d+)*)%?$")


@dataclass(frozen=True, slots=True)
class ParsedNumber:
    raw: str
    value: float | None
    is_missing: bool = False
    is_dash: bool = False
    is_percent: bool = False


def _localized_to_float(text: str) -> float:
    if "." in text and "," in text:
        decimal_sep = "." if text.rfind(".") > text.rfind(",") else ","
        thousands_sep = "," if decimal_sep == "." else "."
        return float(text.replace(thousands_sep, "").replace(decimal_sep, "."))
    if "." in text:
        parts = text.split(".")
        if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3):
            return float("".join(parts))
        return float(text)
    if "," in text:
        # Vietnamese accounting convention uses a comma for decimals.
        return float(text.replace(",", "."))
    return float(text)


def parse_financial_number(raw: object, *, dash_value: float = 0.0) -> ParsedNumber:
    if raw is None:
        return ParsedNumber(raw="", value=None, is_missing=True)
    text = normalize_text(str(raw))
    if not text:
        return ParsedNumber(raw=text, value=None, is_missing=True)
    if text in _DASHES:
        return ParsedNumber(raw=text, value=dash_value, is_dash=True)

    negative_parentheses = text.startswith("(") and text.endswith(")")
    if negative_parentheses:
        text = normalize_text(text[1:-1])
    text = _CURRENCY_RE.sub("", text).replace(" ", "")
    is_percent = text.endswith("%")
    if not _NUMERIC_RE.fullmatch(text):
        return ParsedNumber(raw=str(raw), value=None)
    if is_percent:
        text = text[:-1]
    try:
        value = _localized_to_float(text)
    except ValueError:
        return ParsedNumber(raw=str(raw), value=None)
    if negative_parentheses:
        value = -value
    if not math.isfinite(value):
        return ParsedNumber(raw=str(raw), value=None)
    return ParsedNumber(raw=str(raw), value=value, is_percent=is_percent)


def is_numeric_cell(raw: object) -> bool:
    return parse_financial_number(raw).value is not None
