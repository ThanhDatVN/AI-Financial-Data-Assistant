from __future__ import annotations

from dataclasses import dataclass

from vifinqa.parsing.normalize import ascii_compact


@dataclass(frozen=True, slots=True)
class TargetUnit:
    name: str
    divisor: float
    confidence: float


# Longest/specific patterns must precede their suffixes.
_TARGET_PATTERNS: tuple[tuple[str, float, tuple[str, ...]], ...] = (
    ("TRILLION_VND", 1e12, ("nghintydong",)),
    ("HUNDRED_BILLION_VND", 1e11, ("tramtydong",)),
    ("MILLION_SHARES", 1e6, ("trieucophieu",)),
    ("MILLION_USD", 1e6, ("trieuusd", "millionusd", "trieudolamy")),
    ("BILLION_VND", 1e9, ("tydong", "tyvnd")),
    ("MILLION_VND", 1e6, ("trieudong", "trieuvnd", "maytrieu")),
    ("THOUSAND_VND", 1e3, ("nghindong", "ngan dong", "ngandong", "nghinvnd")),
    ("PERCENT", 1.0, ("phantram", "baonhieuphantram")),
    ("VND", 1.0, ("donvidong", "baonhieudong", "tinhbangdong", "vnd")),
)


def detect_target_unit(question: str) -> TargetUnit:
    compact = ascii_compact(question)
    if "%" in question:
        return TargetUnit("PERCENT", 1.0, 1.0)
    for name, divisor, patterns in _TARGET_PATTERNS:
        if any(pattern in compact for pattern in patterns):
            return TargetUnit(name, divisor, 1.0)
    if "namnao" in compact:
        return TargetUnit("YEAR", 1.0, 0.98)
    if any(marker in compact for marker in ("cophieu", "cophan")):
        return TargetUnit("SHARES", 1.0, 0.98)
    count_markers = (
        "baonhieucongty",
        "baonhieudoanhnghiep",
        "baonhieunganhang",
        "baonhieunam",
        "tongsocongty",
        "sonam",
        "sobaocao",
        "soky",
    )
    if any(marker in compact for marker in count_markers):
        return TargetUnit("COUNT", 1.0, 0.95)
    ratio_markers = (
        "baonhieulan",
        "maylan",
        "baonhieuvong",
        "gapbaonhieu",
        "tinh tyle",
        "tinhtyle",
        "tinh tyso",
        "tinhtyso",
        "tyletrungbinh",
        "tytrong",
    )
    if any(marker in compact for marker in ratio_markers):
        return TargetUnit("RATIO", 1.0, 0.95)
    return TargetUnit("UNKNOWN", 1.0, 0.0)
