from __future__ import annotations

from dataclasses import dataclass

from vifinqa.indexing.manifest import ManifestRecord
from vifinqa.parsing.models import ParsedTable


@dataclass(frozen=True, slots=True)
class CandidateSchema:
    variable: str
    record: ManifestRecord
    table: ParsedTable


def _render_candidate(candidate: CandidateSchema) -> str:
    headers = "\n".join(
        f"  c{index}: {label}" for index, label in enumerate(candidate.table.headers)
    )
    rows = "\n".join(
        f"  r{index}: {row[0] if row else ''}" for index, row in enumerate(candidate.table.rows)
    )
    record = candidate.record
    return (
        f'<table variable="{candidate.variable}" table_ref="{record.table_ref}" '
        f'ticker="{record.ticker}" report_year="{record.report_year}" '
        f'scope="{record.scope}" source_unit="{candidate.table.unit}">\n'
        f"columns:\n{headers}\nrows:\n{rows}\n</table>"
    )


def build_program_prompt(
    question: str,
    candidates: list[CandidateSchema],
    *,
    target_unit: str,
    target_divisor: float,
) -> tuple[str, str]:
    system = """You translate a Vietnamese financial question into a typed arithmetic IR tree.
Each variable is a normalized long pandas DataFrame with columns row_index, column_index,
row_label, column_label, source_unit, raw_value, numeric_value, and base_value. Source numbers are
hidden from you. Select cells only by the supplied row/column coordinates; never copy or invent a
source value.
Use cell nodes with base_value for VND/USD/share amounts and numeric_value for percentages, ratios,
years, or counts. Dimension must describe each operand. Never convert currencies without explicit
exchange-rate evidence. The deterministic compiler applies target_divisor after validating the tree,
so do not add target-unit scaling. Use binary, aggregate, count_if, or arg_extremum nodes as needed.
Return only JSON matching the supplied schema; never emit Python/Pandas code or source values."""
    rendered = "\n\n".join(_render_candidate(candidate) for candidate in candidates)
    user = (
        f"Question: {question}\n"
        f"target_unit={target_unit}; target_divisor={target_divisor}\n\n"
        f"Candidate schemas (labels only; no source values):\n{rendered}\n\n"
        "Choose the minimum evidence variables needed and emit selected_variables plus program."
    )
    return system, user
