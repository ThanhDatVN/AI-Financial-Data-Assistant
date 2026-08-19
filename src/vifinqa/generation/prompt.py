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
    """Where this row holds numbers, and -- said out loud -- when it holds none.

    Rendering an empty row as an empty string leaves the model unable to tell "this row has no
    numbers" from "nothing was said about this row". Five of the 21 failures on the widest
    questions pointed at a heading or a label-only row, which is exactly the mistake the silence
    invites.
    """
    if candidate.numeric_cells is None:
        return ""
    columns = sorted(column for row, column in candidate.numeric_cells if row == row_index)
    if not columns:
        return "  -> NO VALUES; this row is a label, never a cell node"
    return "  -> values at " + ",".join(f"c{column}" for column in columns)


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


# One worked example of the shape a target expects, chosen by that target and nothing else.
#
# Not a style guide -- a measurement. On 499 dev samples with the gold tables already in the
# prompt, every family whose answer is a currency amount scored 0.84 or 0.13, while PERCENT scored
# 0.03 and 0.00 and YEAR scored 0.00. Across three scored runs the model emitted `arg_extremum` 0
# times in 519, 640 and 631 clean programs: the one node that returns a label rather than a value
# has never appeared, and the only place the prompt mentions it describes it as "the same thing in
# one node" without ever showing its fields. `keys` and `values`, not `keys` and `members`.
#
# The project has already paid for this lesson once at the other end of the pipeline: the question
# renderer was told the rule twice and kept writing statements, and only a worked example ending
# in a question mark moved it -- 83% to 100%. Saying the rule is not showing the shape.
#
# Scoped to the asked-for target rather than showing all three, because prompt budget is the
# scarce thing here and the coverage that matters is coverage of the structure THIS answer needs.
# Coordinates are illustrative; value_column and dimension are omitted on cells exactly as the
# system prompt instructs, so the example cannot teach a habit the rest of the prompt forbids.
_WORKED_EXAMPLES: dict[str, str] = {
    "YEAR": (
        "The answer is a year, so it is one of `values`, and `keys` holds the amounts you rank "
        "by. One node does it:\n"
        '{"kind":"arg_extremum","mode":"argmax",'
        '"keys":[{"kind":"cell","variable":"df1","row_index":4,"column_index":2},'
        '{"kind":"cell","variable":"df2","row_index":4,"column_index":2}],'
        '"values":[{"kind":"literal","value":2022,"dimension":"YEAR"},'
        '{"kind":"literal","value":2023,"dimension":"YEAR"}]}'
    ),
    "PERCENT": (
        "A proportion is a division, then times one hundred. Returning either amount on its own "
        "answers a question nobody asked:\n"
        '{"kind":"binary","operator":"*","left":{"kind":"binary","operator":"/",'
        '"left":{"kind":"cell","variable":"df1","row_index":14,"column_index":1},'
        '"right":{"kind":"cell","variable":"df1","row_index":1,"column_index":1}},'
        '"right":{"kind":"literal","value":100,"dimension":"DIMENSIONLESS"}}\n'
        "For a change between two years the numerator is the difference: "
        '{"kind":"binary","operator":"-","left":{"kind":"cell","variable":"df2",'
        '"row_index":4,"column_index":2},"right":{"kind":"cell","variable":"df1",'
        '"row_index":4,"column_index":2}} in place of the single cell above.'
    ),
}
_WORKED_EXAMPLES["RATIO"] = _WORKED_EXAMPLES["PERCENT"]


def worked_example_for(target_unit: str) -> str:
    """The example block for a target, or empty for the amounts that already work."""
    return _WORKED_EXAMPLES.get(target_unit.upper(), "")


def build_program_prompt(
    question: str,
    candidates: list[CandidateSchema],
    *,
    target_unit: str,
    target_divisor: float,
    required_tickers: list[str] | None = None,
    required_years: list[int] | None = None,
    include_row_hierarchy: bool = False,
    worked_example: bool = False,
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
A select node returns one of its `members`, so each member must be the quantity the question asks
you to REPORT, while `keys` is the separate quantity you RANK by. When the question asks for a
percentage or a ratio -- "tỷ lệ ... là bao nhiêu %", "chiếm bao nhiêu phần trăm" -- every member
must itself be that ratio, a binary division of two cells, never the raw amount the ratio is
computed from. Ranking by one quantity and reporting another is the whole point of `keys`: put the
figure you compare in `keys` and the figure you answer with in `members`.
The same rule decides "which year" questions -- "năm nào ... cao nhất", "vào năm nào" -- and it is
the one they get wrong. The answer is a YEAR, so `members` must be the year literals and `keys` the
amounts you rank them by; `arg_extremum` says the same thing in one node. Returning the amount
answers a question nobody asked, and grounding rejects it for reporting VND where a year was
wanted.
When the question asks for one reported figure, answer with exactly one cell node. A statement
repeats the same figure across schedules, often with the opposite sign in a cash-flow adjustment, so
adding those restatements cancels them to zero instead of confirming the figure. Combine cells only
when the question genuinely asks for a total, a difference, a ratio, or a comparison.
selected_variables must list exactly the variables your program reads, and nothing you merely
consulted. Omit value_column and dimension on a cell whose table declares a source unit; grounding
fills both from that unit, and leaving them out keeps a wide cohort program short enough to finish.
Return only JSON matching the supplied schema; never emit Python/Pandas code or source values."""
    example = worked_example_for(target_unit) if worked_example else ""
    if example:
        system += f"\nWorked example for target_unit={target_unit}. {example}"
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
    # The group's size follows from the routes, and the resolver already knows both halves. Left
    # unsaid, question 442 built a 32-member cohort where the routing fans out to 18, then ran out
    # of budget mid-program on all three attempts; question 399 built 32 against 10 and gave its
    # conditions one operand instead of 32. Neither is a reasoning failure -- the model simply had
    # no idea how many members it was supposed to end up with.
    cohort_size = len(required_tickers or []) * len(required_years or [])
    if cohort_size > 1:
        coverage += (
            f"This question routes to {len(required_tickers or [])} tickers across "
            f"{len(required_years or [])} report years, so a select node here has "
            f"{cohort_size} members: one per ticker-year. Every condition's `left` must list "
            f"exactly those same {cohort_size} entries in the same order, and so must `keys`. "
            "Do not enumerate more members than routes; a longer cohort costs the budget the "
            "program needs to finish.\n"
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
