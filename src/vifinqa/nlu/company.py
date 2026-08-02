from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz.fuzz import token_set_ratio
from unidecode import unidecode

from vifinqa.parsing.normalize import ascii_compact, ascii_words


@dataclass(frozen=True, slots=True)
class Company:
    ticker: str
    name: str


@dataclass(frozen=True, slots=True)
class CompanyMatch:
    company: Company
    confidence: float
    method: str


def load_companies(path: Path) -> tuple[Company, ...]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) < 2:
            raise ValueError(f"Company CSV must have ticker and company-name columns: {path}")
        ticker_column, name_column = reader.fieldnames[:2]
        return tuple(
            Company(ticker=row[ticker_column].strip().upper(), name=row[name_column].strip())
            for row in reader
        )


_LEGAL_PREFIXES = (
    "cong ty co phan ",
    "ngan hang thuong mai co phan ",
    "ngan hang tmcp ",
    "tong cong ty co phan ",
    "tong cong ty ",
    "ctcp ",
)


def _token_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", ascii_words(text)).strip()


def _has_title_cased_occurrence(question: str, alias: str) -> bool:
    """Reject accidental two-word verb/place matches while retaining proper names."""
    case_preserving = re.sub(r"[^A-Za-z0-9]+", " ", unidecode(question)).strip()
    pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", re.IGNORECASE)
    for match in pattern.finditer(case_preserving):
        words = match.group(0).split()
        if all(word[0].isupper() for word in words if word):
            return True
    return False


def _company_alias_candidates(name: str) -> set[str]:
    words = _token_text(name)
    candidates = {words}
    if words.startswith("ctcp "):
        candidates.add("cong ty co phan " + words[5:])
    core = words
    changed = True
    while changed:
        changed = False
        for prefix in _LEGAL_PREFIXES:
            if core.startswith(prefix):
                core = core[len(prefix) :]
                changed = True
                break
    core = re.sub(r"\s+(?:ctcp|cong ty co phan)$", "", core).strip()
    candidates.add(core)
    tokens = core.split()
    for width in (2, 3, 4):
        if len(tokens) >= width:
            candidates.add(" ".join(tokens[-width:]))
    return {alias for alias in candidates if len(ascii_compact(alias)) >= 5}


_BRAND_ALIASES: dict[str, str] = {
    "bac a": "BAB",
    "bao viet": "BVH",
    "bidv": "BID",
    "binh son": "BSR",
    "dabaco": "DBC",
    "dam ca mau": "DCM",
    "dam phu my": "DPM",
    "dat xanh": "DXG",
    "eximbank": "EIB",
    "hoa phat": "HPG",
    "hoa sen": "HSG",
    "mbbank": "MBB",
    "nam kim": "NKG",
    "novaland": "NVL",
    "quan doi": "MBB",
    "sabeco": "SAB",
    "saigonbank": "SGB",
    "tkv": "DTK",
    "vietcombank": "VCB",
    "vietinbank": "CTG",
    "vinamilk": "VNM",
    "vingroup": "VIC",
}

_GENERIC_ALIASES = {
    "bat dong san",
    "chung khoan",
    "dau khi",
    "dich vu",
    "dien luc",
    "ngan hang",
    "tai chinh",
    "thuong mai",
    "xay dung",
}


class CompanyResolver:
    def __init__(self, companies: tuple[Company, ...]) -> None:
        self.companies = companies
        aliases: dict[str, list[Company]] = {}
        for company in companies:
            for alias in _company_alias_candidates(company.name):
                aliases.setdefault(alias, []).append(company)
        # Short aliases are accepted only when they identify exactly one company in this corpus.
        self.unique_aliases = {
            alias: matches[0]
            for alias, matches in aliases.items()
            if len(matches) == 1 and alias not in _GENERIC_ALIASES
        }
        by_ticker = {company.ticker: company for company in companies}
        for alias, ticker in _BRAND_ALIASES.items():
            if ticker in by_ticker:
                self.unique_aliases[_token_text(alias)] = by_ticker[ticker]

    @classmethod
    def from_csv(cls, path: Path) -> CompanyResolver:
        return cls(load_companies(path))

    def resolve(self, question: str) -> tuple[CompanyMatch, ...]:
        question_tokens = _token_text(question)
        contained_by_ticker: dict[str, tuple[int, int, Company]] = {}
        for alias, company in self.unique_aliases.items():
            if (
                len(alias.split()) == 2
                and alias not in _BRAND_ALIASES
                and not _has_title_cased_occurrence(question, alias)
            ):
                continue
            position = f" {question_tokens} ".find(f" {alias} ")
            if position < 0:
                continue
            prior = contained_by_ticker.get(company.ticker)
            if prior is None or len(alias) > prior[0]:
                contained_by_ticker[company.ticker] = (len(alias), position, company)
        alias_matches = list(contained_by_ticker.values())
        explicit: list[CompanyMatch] = []
        for company in self.companies:
            ticker_token = company.ticker.lower()
            for ticker_match in re.finditer(
                rf"(?<![a-z0-9]){re.escape(ticker_token)}(?![a-z0-9])", question_tokens
            ):
                start, end = ticker_match.span()
                shadowed = any(
                    other.ticker != company.ticker
                    and length > len(ticker_token)
                    and position <= start
                    and end <= position + length
                    for length, position, other in alias_matches
                )
                if not shadowed:
                    explicit.append(CompanyMatch(company, 1.0, "ticker"))
                    break
        contained = [
            CompanyMatch(company, 0.98, "unique_alias")
            for length, position, company in sorted(alias_matches, key=lambda item: item[1])
            if not any(
                other_length > length
                and other_position <= position
                and position + length <= other_position + other_length
                for other_length, other_position, _ in alias_matches
            )
        ]
        combined = {match.company.ticker: match for match in explicit}
        for match in contained:
            combined.setdefault(match.company.ticker, match)
        if combined:
            return tuple(combined.values())

        normalized_question = question_tokens
        scored = sorted(
            (
                (token_set_ratio(normalized_question, ascii_words(company.name)), company)
                for company in self.companies
            ),
            reverse=True,
            key=lambda item: item[0],
        )
        if not scored:
            return ()
        best_score, best = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        # A fuzzy fallback is emitted only with both high absolute score and a clear margin.
        if best_score >= 88.0 and best_score - second_score >= 5.0:
            return (CompanyMatch(best, best_score / 100.0, "fuzzy"),)
        return ()
