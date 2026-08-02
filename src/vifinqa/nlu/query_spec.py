from __future__ import annotations

import re
from dataclasses import dataclass

from vifinqa.nlu.company import CompanyMatch, CompanyResolver
from vifinqa.nlu.scope import detect_scope_intent
from vifinqa.nlu.temporal import extract_temporal_mentions
from vifinqa.nlu.unit_target import detect_target_unit
from vifinqa.parsing.normalize import ascii_words


@dataclass(frozen=True, slots=True)
class EntitySpec:
    ticker: str
    scope: str | None
    confidence: float


@dataclass(frozen=True, slots=True)
class TemporalSpec:
    year: int
    basis: str
    report_year: int
    column_role: str


@dataclass(frozen=True, slots=True)
class QuerySpec:
    question: str
    entities: tuple[EntitySpec, ...]
    periods: tuple[TemporalSpec, ...]
    target_unit: str
    target_divisor: float


def _select_target_matches(question: str, resolver: CompanyResolver) -> tuple[CompanyMatch, ...]:
    matches = resolver.resolve(question)
    if len(matches) <= 1:
        return matches
    normalized = ascii_words(question)

    parent_marker = "cua cong ty me "
    marker_index = normalized.find(parent_marker)
    if marker_index >= 0 and normalized.count("cong ty me ") == 1:
        tail = normalized[marker_index + len(parent_marker) :]
        stop = re.search(r"\b(?:vao|tai|cuoi|dau|trong)\b|\bnam\s+20\d{2}\b", tail)
        segment = tail[: stop.start()] if stop else tail
        primary = resolver.resolve(segment)
        if len(primary) == 1 and any(
            match.company.ticker == primary[0].company.ticker for match in matches
        ):
            return primary

    ownership = re.search(r"\bcua\s+([^,]+),", normalized)
    counterparty_clause = (
        re.search(
            r"\b(?:ban hang|giao dich|mua hang|phai tra|phai thu)\b" r".{0,50}\b(?:voi|cho|tu)\b",
            normalized[ownership.end() :],
        )
        if ownership
        else None
    )
    if ownership and counterparty_clause:
        primary = resolver.resolve(ownership.group(1))
        if len(primary) == 1 and any(
            match.company.ticker == primary[0].company.ticker for match in matches
        ):
            return primary
    return matches


def parse_query_spec(question: str, resolver: CompanyResolver) -> QuerySpec:
    scope = detect_scope_intent(question)
    entities = tuple(
        EntitySpec(match.company.ticker, scope.scope, match.confidence)
        for match in _select_target_matches(question, resolver)
    )
    periods = tuple(
        TemporalSpec(item.year, item.basis, item.preferred_report_year, item.column_role)
        for item in extract_temporal_mentions(question)
    )
    target = detect_target_unit(question)
    return QuerySpec(question, entities, periods, target.name, target.divisor)
