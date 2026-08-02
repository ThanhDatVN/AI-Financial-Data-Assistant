from __future__ import annotations

import re
from dataclasses import dataclass

from vifinqa.parsing.normalize import ascii_compact, ascii_words

_YEAR_RE = re.compile(r"(?<!\d)(20(?:1[5-9]|2[0-6]))(?!\d)")


@dataclass(frozen=True, slots=True)
class TemporalMention:
    year: int
    basis: str
    preferred_report_year: int
    column_role: str


def extract_temporal_mentions(question: str) -> tuple[TemporalMention, ...]:
    normalized = ascii_words(question)
    compact = ascii_compact(question)
    mentions: list[TemporalMention] = []
    year_matches = list(_YEAR_RE.finditer(normalized))
    for match in year_matches:
        year = int(match.group(1))
        before = normalized[max(0, match.start() - 28) : match.start()]
        around = normalized[max(0, match.start() - 28) : match.end() + 12]
        start_of_year = bool(re.search(r"(?:dau nam|dau ky)\s*$", before))
        end_of_year = bool(
            re.search(r"(?:cuoi nam|cuoi ky)\s*$", before)
            or ("den ngay" in around and re.search(r"\d{1,2}[/-]\d{1,2}[/-]$", before))
        )
        both_roles = f"daunamdencuoinam{year}" in compact or f"daunamvacuoinam{year}" in compact
        if both_roles:
            mentions.append(TemporalMention(year, "point_in_time", year, "prior_period"))
            mentions.append(TemporalMention(year, "point_in_time", year, "current_period"))
        elif start_of_year:
            mentions.append(TemporalMention(year, "point_in_time", year, "prior_period"))
        elif end_of_year:
            mentions.append(TemporalMention(year, "point_in_time", year, "current_period"))
        else:
            mentions.append(TemporalMention(year, "flow_or_unspecified", year, "current_period"))
    for left, right in zip(year_matches, year_matches[1:], strict=False):
        start_year = int(left.group(1))
        end_year = int(right.group(1))
        between = normalized[left.end() : right.start()]
        is_range = bool(re.search(r"[-–—]", between) or re.search(r"\b(?:den|toi)\b", between))
        if is_range and 1 < end_year - start_year <= 10:
            mentions.extend(
                TemporalMention(year, "range", year, "current_period")
                for year in range(start_year + 1, end_year)
            )
    return tuple(dict.fromkeys(mentions))
