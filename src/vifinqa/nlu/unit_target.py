from __future__ import annotations

from dataclasses import dataclass

from vifinqa.parsing.normalize import ascii_compact, ascii_words


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


def _match_explicit_unit(compact: str) -> TargetUnit | None:
    for name, divisor, patterns in _TARGET_PATTERNS:
        if any(pattern in compact for pattern in patterns):
            return TargetUnit(name, divisor, 1.0)
    return None


def _target_tail(question: str) -> str | None:
    normalized = ascii_words(question)
    marker = "bao nhieu"
    marker_index = normalized.rfind(marker)
    return ascii_compact(normalized[marker_index:]) if marker_index >= 0 else None


def _detect_semantic_unit(compact: str) -> TargetUnit | None:
    if "namnao" in compact:
        return TargetUnit("YEAR", 1.0, 0.98)
    if any(marker in compact for marker in ("baonhieucophieu", "baonhieucophan")):
        return TargetUnit("SHARES", 1.0, 0.98)
    count_markers = (
        "baonhieucongty",
        "baonhieudoanhnghiep",
        "baonhieunganhang",
        "baonhieudonvi",
        "baonhieuma",
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
        "tinhtyle",
        "tinhtyso",
        "tyletrungbinh",
        "tytrong",
    )
    if any(marker in compact for marker in ratio_markers):
        return TargetUnit("RATIO", 1.0, 0.95)
    return None


def detect_target_unit(question: str) -> TargetUnit:
    compact = ascii_compact(question)
    tail = _target_tail(question)
    if tail is not None:
        explicit_tail = _match_explicit_unit(tail)
        if explicit_tail is not None:
            return explicit_tail
        semantic_tail = _detect_semantic_unit(tail)
        if semantic_tail is not None:
            return semantic_tail

    semantic = _detect_semantic_unit(compact)
    if semantic is not None and semantic.name == "YEAR":
        return semantic
    if "%" in question:
        return TargetUnit("PERCENT", 1.0, 1.0)
    explicit = _match_explicit_unit(compact)
    if explicit is not None:
        return explicit
    if semantic is not None:
        return semantic
    if any(marker in compact for marker in ("cophieu", "cophan")):
        return TargetUnit("SHARES", 1.0, 0.98)
    return TargetUnit("UNKNOWN", 1.0, 0.0)
