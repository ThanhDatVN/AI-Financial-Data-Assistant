from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from vifinqa.indexing.manifest import ManifestRecord
from vifinqa.parsing.models import ParsedTable


@dataclass(frozen=True, slots=True)
class CandidateSchema:
    variable: str
    record: ManifestRecord
    table: ParsedTable
    # Which coordinates actually hold a parsed number. Roughly two cells in five are blank,
    # dashes or labels, and without this the model picks coordinates blind and loses the
    # question to a cell that has no value. The positions are structure, not source values.
    numeric_cells: frozenset[tuple[int, int]] | None = None


def numeric_cells_of(frame: pd.DataFrame) -> frozenset[tuple[int, int]]:
    if not {"row_index", "column_index", "numeric_value"} <= set(frame.columns):
        return frozenset()
    populated = frame.loc[frame["numeric_value"].notna(), ["row_index", "column_index"]]
    return frozenset((int(row), int(column)) for row, column in populated.itertuples(index=False))


def _numeric_columns(candidate: CandidateSchema, row_index: int) -> str:
    if candidate.numeric_cells is None:
        return ""
    columns = sorted(column for row, column in candidate.numeric_cells if row == row_index)
    return "  -> values at " + ",".join(f"c{column}" for column in columns) if columns else ""


_LETTERED_ROW_RE = re.compile(r"^\s*([A-ZĐ]+)\s*[.)]\s+", flags=re.IGNORECASE)
_NUMBERED_ROW_RE = re.compile(r"^\s*\d+(?:\.\d+)*\s*[.)]\s+")
_BULLETED_ROW_RE = re.compile(r"^\s*[-+•▪●○◦]\s+")


def _row_level(label: str) -> int | None:
    """Infer only explicit accounting-outline levels; ambiguous plain text stays flat."""
    lettered = _LETTERED_ROW_RE.match(label)
    if lettered:
        token = lettered.group(1).upper()
        # Vietnamese statements conventionally use A/B/... for top-level groups and
        # I/II/III/... for the next level. Treat only common I/V/X combinations as Roman;
        # a single C or D is much more likely a lettered group than 100 or 500.
        return 1 if set(token) <= {"I", "V", "X"} else 0
    if _NUMBERED_ROW_RE.match(label):
        return 2
    if _BULLETED_ROW_RE.match(label):
        return 3
    return None


def _row_hierarchy_contexts(rows: tuple[tuple[str, ...], ...]) -> tuple[str, ...]:
    """Render conservative ancestor paths without changing source row coordinates."""
    ancestors: dict[int, str] = {}
    contexts: list[str] = []
    for row in rows:
        label = row[0].strip() if row else ""
        populated = [cell.strip() for cell in row if cell.strip()]
        # OCR often represents a colspan section title by repeating the same text in every
        # column. It is safe to treat that exact structural signal as the parent of bullets.
        if label and len(populated) >= 2 and all(cell == label for cell in populated):
            ancestors = {2: label}
            contexts.append("")
            continue
        level = _row_level(label)
        if level is None:
            contexts.append("")
            continue
        parents = [ancestors[index] for index in sorted(ancestors) if index < level]
        contexts.append(" [parents: " + " > ".join(parents) + "]" if parents else "")
        ancestors = {index: value for index, value in ancestors.items() if index < level}
        ancestors[level] = label
    return tuple(contexts)


def _render_candidate(candidate: CandidateSchema, *, include_row_hierarchy: bool) -> str:
    headers = "\n".join(
        f"  c{index}: {label}" for index, label in enumerate(candidate.table.headers)
    )
    hierarchy_contexts = (
        _row_hierarchy_contexts(candidate.table.rows)
        if include_row_hierarchy
        else ("",) * len(candidate.table.rows)
    )
    rows = "\n".join(
        f"  r{index}: {row[0] if row else ''}{hierarchy_contexts[index]}"
        f"{_numeric_columns(candidate, index)}"
        for index, row in enumerate(candidate.table.rows)
    )
    record = candidate.record
    # The section a table sits in is what separates a reported figure from the same number
    # restated as an adjustment elsewhere in the statement, so the model has to see it.
    section = f' section="{record.section_title}"' if record.section_title else ""
    return (
        f'<table variable="{candidate.variable}" table_ref="{record.table_ref}" '
        f'ticker="{record.ticker}" report_year="{record.report_year}" '
        f'scope="{record.scope}"{section} source_unit="{candidate.table.unit}">\n'
        f"columns:\n{headers}\nrows:\n{rows}\n</table>"
    )


def build_program_prompt(
    question: str,
    candidates: list[CandidateSchema],
    *,
    target_unit: str,
    target_divisor: float,
    required_tickers: list[str] | None = None,
    required_years: list[int] | None = None,
    include_row_hierarchy: bool = False,
) -> tuple[str, str]:
    system = """You translate a Vietnamese financial question into a typed arithmetic IR tree.
Each variable is a normalized long pandas DataFrame with columns row_index, column_index,
row_label, column_label, source_unit, raw_value, numeric_value, and base_value. Source numbers are
hidden from you. Select cells only by the supplied row/column coordinates; never copy or invent a
source value.
Use cell nodes with base_value for VND/USD/share amounts and numeric_value for percentages, ratios,
years, or counts. Dimension must describe each operand. Never convert currencies without explicit
exchange-rate evidence. The deterministic compiler applies target_divisor after validating the tree,
so do not add target-unit scaling. Use binary, aggregate, count_if or arg_extremum as needed.
When a question asks about the members of a group that satisfy something, such as the companies
whose margin stayed positive or the half of a group below the median, use a select node: one
member expression per company, one condition per test with one entry per company, and an
operator among sum, mean, min, max, count, argmin and argmax. A median is a select node with no
conditions at all.
When the question asks for one reported figure, answer with exactly one cell node. A statement
repeats the same figure across schedules, often with the opposite sign in a cash-flow adjustment, so
adding those restatements cancels them to zero instead of confirming the figure. Combine cells only
when the question genuinely asks for a total, a difference, a ratio, or a comparison.
selected_variables must list exactly the variables your program reads, and nothing you merely
consulted. Omit value_column and dimension on a cell whose table declares a source unit; grounding
fills both from that unit, and leaving them out keeps a wide cohort program short enough to finish.
Return only JSON matching the supplied schema; never emit Python/Pandas code or source values."""
    rendered = "\n\n".join(
        _render_candidate(candidate, include_row_hierarchy=include_row_hierarchy)
        for candidate in candidates
    )
    # A cohort question names its group in Vietnamese prose, and a program that answers for
    # one member of the group is rejected. The resolver already extracted the group, so state
    # it as a requirement rather than making the model recover it from the sentence.
    coverage = ""
    if required_tickers:
        coverage += (
            f"Your program must read at least one cell from every one of these tickers: "
            f"{', '.join(required_tickers)}.\n"
        )
    if required_years:
        coverage += (
            "It must also cover every one of these report years: "
            f"{', '.join(str(year) for year in required_years)}.\n"
        )
    hierarchy_note = (
        " Explicit accounting outlines also show `parents:` to disambiguate repeated child "
        "labels; parents are context, not extra rows."
        if include_row_hierarchy
        else ""
    )
    user = (
        f"Question: {question}\n"
        f"target_unit={target_unit}; target_divisor={target_divisor}\n"
        f"{coverage}\n"
        f"Candidate schemas (labels only; no source values):\n{rendered}\n\n"
        "Rows list the coordinates that hold a number as `-> values at c...`; a cell node is "
        f"only valid at one of those coordinates.{hierarchy_note}\n"
        "Choose the minimum evidence variables needed and emit selected_variables plus program."
    )
    return system, user
