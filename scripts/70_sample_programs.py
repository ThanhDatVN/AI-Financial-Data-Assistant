"""Build labelled training data backwards, from tables the corpus already holds.

The project cannot measure itself. Every question in `annotations/qrels_template.jsonl` is still
`unlabeled`, so "about 24% of the programs the model writes are correct" remains an estimate
drawn from an assumption, and every change to the answer branch costs twelve GPU hours plus a
submission before anyone learns whether it helped.

Going forwards -- question first, then program -- would need the model to write the program, and
that is the very thing under measurement. Going backwards does not: sample a program over a real
table, execute it, and the answer is right by construction. A separate step renders the Vietnamese
question afterwards, which is the one job an open model does reliably.

Nothing here calls a model. Nothing here needs a GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, NamedTuple, cast

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vifinqa.evidence.store import TableStore, parsed_table_to_long_frame  # noqa: E402
from vifinqa.indexing.manifest import ManifestRecord  # noqa: E402
from vifinqa.programs.compiler import compile_expression  # noqa: E402
from vifinqa.programs.executor import execute_expression  # noqa: E402
from vifinqa.programs.grounding import (  # noqa: E402
    TARGET_DIMENSIONS,
    prepare_program,
    referenced_variables,
    validate_answer_plausibility,
)
from vifinqa.programs.ir import (  # noqa: E402
    ArgExtremumExpr,
    BinaryExpr,
    CellExpr,
    Condition,
    Dimension,
    LiteralExpr,
    ScalarExpr,
    SelectExpr,
)
from vifinqa.programs.serde import expression_to_dict  # noqa: E402

# The question asks for a unit, and the program has to land in it. Two names are needed, not one:
# the renderer writes the Vietnamese phrase into the question, while grounding and the solver
# prompt speak the enum production uses. Recording only the phrase left every sample unusable by
# `prepare_program`, which knows BILLION_VND and has never heard of "tỷ đồng".
TARGET_UNITS: tuple[tuple[str, str, float], ...] = (
    ("đồng", "VND", 1.0),
    ("triệu đồng", "MILLION_VND", 1e6),
    ("tỷ đồng", "BILLION_VND", 1e9),
    ("trăm tỷ đồng", "HUNDRED_BILLION_VND", 1e11),
)
PERCENT_UNIT = ("phần trăm", "PERCENT", 1.0)
YEAR_UNIT = ("năm", "YEAR", 1.0)

# A cell's source unit fixes what it measures, and grounding will overwrite any other claim with
# it. Asking for a figure in "tỷ đồng" when the table counts shares is not a hard question, it is
# an impossible one: `prepare_program` rejected every such draw, and the sampler was picking a
# money unit for every cell regardless of what its table held.
_UNITS_BY_DIMENSION: dict[str, tuple[tuple[str, str, float], ...]] = {
    "VND": TARGET_UNITS,
    "SHARES": (("cổ phiếu", "SHARES", 1.0), ("triệu cổ phiếu", "MILLION_SHARES", 1e6)),
    "USD": (("triệu USD", "MILLION_USD", 1e6),),
    "PERCENT": (PERCENT_UNIT,),
}
# What the store writes into `source_unit`, mapped the way grounding maps it.
_SOURCE_DIMENSIONS = {
    "VND": "VND",
    "THOUSAND_VND": "VND",
    "MILLION_VND": "VND",
    "BILLION_VND": "VND",
    "USD": "USD",
    "MILLION_USD": "USD",
    "PERCENT": "PERCENT",
    "SHARES": "SHARES",
}

# A unit the grounding layer does not recognise fails every sample that draws it, and it fails at
# filter time -- long after this run is over and on a machine with a GPU attached.
for _dimension, _units in (*_UNITS_BY_DIMENSION.items(), ("YEAR", (YEAR_UNIT,))):
    for _phrase, _enum, _divisor in _units:
        if TARGET_DIMENSIONS.get(_enum) != _dimension:
            raise SystemExit(f"target unit {_enum!r} does not measure {_dimension} to grounding")

# Questions about one report far outnumber the rest, so single-table families dominate. The
# organisers publish the real mix (docs/17): Easy 35.7%, Medium 23.2%, Intermediate 19.8%, Hard
# 21.3%. The previous weights were inferred from fan-out and had no Hard tier at all, so a fifth
# of the paper -- and the fifth every model scores worst on -- was missing from a set meant to
# stand in for it.
#
# Module level rather than inside `main` so a test can check that every family this can draw is
# one `71_render_questions.py` knows how to phrase. `conditional` was drawn for months without a
# brief there, which would have written all 106 Hard dev samples as plain extremums.
FAMILY_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("lookup", 0.357),  # Easy
    ("ratio", 0.116),  # Medium, split with change
    ("change", 0.116),  # Medium
    ("extremum", 0.198),  # Intermediate
    ("conditional", 0.213),  # Hard: multi-hop dependent
)

# A label that names no line item cannot anchor a question.
MIN_LABEL_CHARS = 12
# Base values are in dong. A denominator under a billion makes any ratio an artefact of rounding:
# the first smoke run produced a year-on-year change of 24.289.948% from a denominator of a few
# thousand dong.
MIN_DENOMINATOR = 1e9
# No real financial ratio or growth rate in this corpus runs past this, so anything that does is
# a parsing artefact rather than a fact worth asking about.
MAX_ABS_PERCENT = 500.0
# The largest balance sheet in this corpus totals around 2e15 dong, so a cell past 1e16 is not a
# figure. The ones that appear are rows naming a subsidiary in a related-party table, where OCR
# ran adjacent numbers together: the first dev set drew four of them, at 1.5e19 up to 2.1e78.
# Real values stop below 1e15 and the artefacts start above 1e19, so the cut is unambiguous.
MAX_ABS_BASE_VALUE = 1e16

# A row whose label opens with a bullet, a dash or a lone digit is a sub-item: it means nothing
# without the parent row above it, and a question naming it alone has no single answer.
_SUBITEM_PREFIXES = ("-", "+", "*", "•", "–", "(")
# Labels that name a movement line in an equity or provision roll-forward. They repeat across
# every column of the table, so they cannot identify one figure.
_ROLLFORWARD_MARKERS = ("số dư", "số cuối", "số đầu", "tại ngày", "cộng", "tổng cộng phát sinh")


def _normalise(label: str) -> str:
    """Fold a row label to something comparable across report years."""
    folded = unicodedata.normalize("NFKC", label).casefold().strip()
    return " ".join("".join(c for c in folded if c.isalnum() or c.isspace()).split())


class _Draw(NamedTuple):
    """What one draw yields.

    The divisor rides alongside the tree rather than inside it. Production forbids the model from
    writing target-unit scaling into a program -- the compiler applies it afterwards -- so a gold
    program carrying its own division would teach the opposite of the convention it exists to show.

    `condition` is what makes a Hard sample answerable. Its program reads one line item to decide
    which years count and a different one for the answer, and the sampler throws away any draw
    where the two agree. So a question written from `row_label` alone asks a plainer question with
    a provably different answer, and every Hard sample would be rendered unanswerable and then
    filtered out for being unanswered -- losing the 21.3% tier the set exists to cover, while
    looking like the model merely found it hard.
    """

    expression: ScalarExpr
    target_unit: str
    target_unit_text: str
    target_divisor: float
    row_label: str
    column_label: str
    condition: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class Sample:
    """One labelled example: the program, what it computed, and where it came from."""

    family: str
    table_refs: tuple[str, ...]
    doc_ids: tuple[str, ...]
    ticker: str
    report_years: tuple[int, ...]
    scope: str
    row_label: str
    column_label: str
    section_title: str | None
    target_unit: str
    target_unit_text: str
    target_divisor: float
    answer: float
    pandas_query: str
    program: dict[str, object]
    # Only the Hard family has one, and without it that family cannot be phrased. See `_Draw`.
    condition: dict[str, object] | None = None


def _is_askable(label: str) -> bool:
    """Can a question name this row on its own and mean exactly one figure?"""
    stripped = label.strip()
    if len(stripped) < MIN_LABEL_CHARS or stripped.startswith(_SUBITEM_PREFIXES):
        return False
    folded = _normalise(stripped)
    if not folded or folded[0].isdigit():
        return False
    return not any(marker in folded for marker in _ROLLFORWARD_MARKERS)


def _current_period(frame: pd.DataFrame, report_year: int) -> pd.DataFrame:
    """Keep the columns that report the year the document is for.

    A statement carries the prior period beside the current one. Asking "in 2024" and reading the
    31.12.2023 column is a silently wrong label, and the first smoke run did exactly that.
    """
    header = frame["column_label"].astype(str)
    current = frame[header.str.contains(str(report_year), regex=False)]
    if current.empty:
        return current
    # Where several columns name the year, the leftmost is the primary one.
    return current[current["column_index"] == current["column_index"].min()]


def _numeric_cells(frame: pd.DataFrame, *, report_year: int | None = None) -> pd.DataFrame:
    """Rows a program can point at: a real number, under a label worth asking about."""
    # `base_value` holds None for unparsed cells, so the column is object dtype and `.abs()`
    # raises on it. Drop the blanks before doing any arithmetic.
    usable = frame[frame["base_value"].notna() & (frame["column_index"] > 0)]
    if usable.empty:
        return usable
    magnitude = usable["base_value"].astype(float).abs()
    usable = usable[(magnitude > 0) & (magnitude <= MAX_ABS_BASE_VALUE)]
    if usable.empty:
        return usable
    # `.map` over an empty frame returns an object-dtype series, and pandas reads an object-dtype
    # key as a list of column names rather than a mask -- which silently drops every column and
    # turns the next lookup into a KeyError. Casting to bool keeps it a mask either way.
    askable = usable["row_label"].astype(str).map(_is_askable).astype(bool)
    usable = usable[askable]
    if usable.empty or report_year is None:
        return usable
    return _current_period(usable, report_year)


def _execute(
    expression: ScalarExpr, frames: dict[str, pd.DataFrame], unit_name: str, divisor: float
) -> tuple[str, float] | None:
    """Run the program down production's own path, or reject the sample.

    Not a re-implementation of that path: `prepare_program` is the function the solver's output
    goes through, and it is what settles each cell's unit, checks the program's dimension against
    the unit asked for, and applies the divisor. Anything it refuses is a sample no solver could
    ever be scored right on, so refusing it here is the whole point.

    Running the two separately let three faults through unseen at once -- year literals with no
    dimension, thresholds that did not measure what they were compared against, and money units
    asked of tables counting shares. Taking the same path leaves nowhere for a fourth to hide:
    a hand-written check of these samples went from 36% reproducible to 100%.
    """
    selected = sorted(referenced_variables(expression))
    try:
        prepared, _ = prepare_program(
            expression,
            selected_variables=selected,
            frames={name: frames[name] for name in selected},
            target_unit=unit_name,
            target_divisor=divisor,
        )
        query = compile_expression(prepared)
        answer = execute_expression(query, frames)
    except Exception:  # noqa: BLE001 - a sample that will not run is simply not a sample
        return None
    if not math.isfinite(answer):
        return None
    if unit_name == "PERCENT" and abs(answer) > MAX_ABS_PERCENT:
        return None
    try:
        validate_answer_plausibility(answer, expression)
    except Exception:  # noqa: BLE001
        return None
    return query, answer


def _cell_dimension(cell: pd.Series) -> str | None:
    """What this cell measures, read the way grounding reads it, or None when nothing says."""
    return _SOURCE_DIMENSIONS.get(str(cell.get("source_unit") or ""))


def _askable_unit(cell: pd.Series, rng: random.Random) -> tuple[str, str, float] | None:
    """A target unit the cell can actually be reported in, or None if there is none.

    Grounding settles a cell's dimension from its source unit and then insists the target agrees.
    A cell with no declared unit keeps whatever the program claims, and a bare currency claim with
    no lineage behind it is refused later -- so an unlabelled cell is not a lookup worth drawing.
    """
    dimension = _cell_dimension(cell)
    if dimension is None:
        return None
    choices = _UNITS_BY_DIMENSION[dimension]
    return choices[rng.randrange(len(choices))]


def _sample_lookup(frame: pd.DataFrame, record: ManifestRecord, rng: random.Random) -> _Draw | None:
    """F1 -- read one cell and convert it into the unit the question will ask for."""
    usable = _numeric_cells(frame, report_year=int(record.report_year))
    if usable.empty:
        return None
    cell = usable.iloc[rng.randrange(len(usable))]
    chosen = _askable_unit(cell, rng)
    if chosen is None:
        return None
    unit_text, unit_name, divisor = chosen
    read = CellExpr(
        variable="df1",
        row_index=int(cell["row_index"]),
        column_index=int(cell["column_index"]),
    )
    return _Draw(
        read, unit_name, unit_text, divisor, str(cell["row_label"]), str(cell["column_label"])
    )


def _sample_ratio(frame: pd.DataFrame, record: ManifestRecord, rng: random.Random) -> _Draw | None:
    """F2 -- one line item as a percentage of another in the same column."""
    usable = _numeric_cells(frame, report_year=int(record.report_year))
    if len(usable) < 2:
        return None
    column = int(usable.iloc[rng.randrange(len(usable))]["column_index"])
    same_column = usable[usable["column_index"] == column]
    if len(same_column) < 2:
        return None
    # The denominator should be the larger figure, or the ratio is not a share of anything.
    ordered = same_column.reindex(same_column["base_value"].abs().sort_values().index)
    numerator = ordered.iloc[rng.randrange(len(ordered) - 1)]
    denominator = ordered.iloc[-1]
    if abs(float(denominator["base_value"])) < MIN_DENOMINATOR:
        return None
    if _normalise(str(numerator["row_label"])) == _normalise(str(denominator["row_label"])):
        return None
    # A quotient of two cells whose unit nothing declares infers UNKNOWN, and no percentage
    # target accepts that. The ratio needs lineage on both halves, not just a number on each.
    if _cell_dimension(numerator) is None or _cell_dimension(denominator) is None:
        return None
    expression = BinaryExpr(
        operator="*",
        left=BinaryExpr(
            operator="/",
            left=CellExpr(
                variable="df1",
                row_index=int(numerator["row_index"]),
                column_index=int(numerator["column_index"]),
            ),
            right=CellExpr(
                variable="df1",
                row_index=int(denominator["row_index"]),
                column_index=int(denominator["column_index"]),
            ),
        ),
        right=LiteralExpr(value=100.0),
    )
    text, name, divisor = PERCENT_UNIT
    return _Draw(
        expression,
        name,
        text,
        divisor,
        str(numerator["row_label"]),
        str(numerator["column_label"]),
    )


def _shared_label_rows(
    frames: dict[str, pd.DataFrame], rng: random.Random
) -> tuple[str, dict[str, pd.Series]] | None:
    """Find one line item that every year's table reports, matched on a folded label."""
    by_label: dict[str, dict[str, pd.Series]] = defaultdict(dict)
    for variable, frame in frames.items():
        year = int(frame["report_year"].iloc[0]) if not frame.empty else None
        usable = _numeric_cells(frame, report_year=year)
        for _, row in usable.iterrows():
            key = _normalise(str(row["row_label"]))
            # First column wins: later columns are prior-period restatements.
            if key and variable not in by_label[key]:
                by_label[key][variable] = row
    shared = [key for key, found in by_label.items() if len(found) == len(frames)]
    if not shared:
        return None
    chosen = shared[rng.randrange(len(shared))]
    return chosen, by_label[chosen]


def _shared_label_pairs(
    frames: dict[str, pd.DataFrame], rng: random.Random
) -> tuple[dict[str, pd.Series], dict[str, pd.Series]] | None:
    """Two different line items that every year reports, for a filter and a separate answer."""
    by_label: dict[str, dict[str, pd.Series]] = defaultdict(dict)
    for variable, frame in frames.items():
        year = int(frame["report_year"].iloc[0]) if not frame.empty else None
        for _, row in _numeric_cells(frame, report_year=year).iterrows():
            key = _normalise(str(row["row_label"]))
            if key and variable not in by_label[key]:
                by_label[key][variable] = row
    shared = [key for key, found in by_label.items() if len(found) == len(frames)]
    if len(shared) < 2:
        return None
    first, second = rng.sample(shared, 2)
    return by_label[first], by_label[second]


def _sample_change(frames: dict[str, pd.DataFrame], rng: random.Random) -> _Draw | None:
    """F3 -- percentage change of one line item between two report years."""
    found = _shared_label_rows(frames, rng)
    if found is None:
        return None
    _, rows = found
    earlier, later = (rows[name] for name in sorted(rows))
    if abs(float(earlier["base_value"])) < MIN_DENOMINATOR:
        return None
    # Same reason as the ratio family: a change is a quotient, and an undeclared unit makes it
    # incomparable with the percentage it is supposed to be reported as.
    if _cell_dimension(earlier) is None or _cell_dimension(later) is None:
        return None
    first = CellExpr(
        variable=str(earlier["variable"]),
        row_index=int(earlier["row_index"]),
        column_index=int(earlier["column_index"]),
    )
    second = CellExpr(
        variable=str(later["variable"]),
        row_index=int(later["row_index"]),
        column_index=int(later["column_index"]),
    )
    expression = BinaryExpr(
        operator="*",
        left=BinaryExpr(
            operator="/", left=BinaryExpr(operator="-", left=second, right=first), right=first
        ),
        right=LiteralExpr(value=100.0),
    )
    text, name, divisor = PERCENT_UNIT
    return _Draw(
        expression, name, text, divisor, str(later["row_label"]), str(later["column_label"])
    )


def _sample_extremum(frames: dict[str, pd.DataFrame], rng: random.Random) -> _Draw | None:
    """F4 -- which year reported the largest (or smallest) value of one line item."""
    found = _shared_label_rows(frames, rng)
    if found is None:
        return None
    _, rows = found
    ordered = sorted(rows)
    values = tuple(
        CellExpr(
            variable=str(rows[name]["variable"]),
            row_index=int(rows[name]["row_index"]),
            column_index=int(rows[name]["column_index"]),
        )
        for name in ordered
    )
    years = tuple(
        LiteralExpr(value=float(rows[name]["report_year"]), dimension="YEAR") for name in ordered
    )
    mode: Literal["argmin", "argmax"] = "argmax" if rng.random() < 0.5 else "argmin"
    expression = ArgExtremumExpr(mode=mode, keys=values, values=years)
    anchor = rows[ordered[-1]]
    text, name, divisor = YEAR_UNIT
    return _Draw(
        expression, name, text, divisor, str(anchor["row_label"]), str(anchor["column_label"])
    )


def _sample_conditional(frames: dict[str, pd.DataFrame], rng: random.Random) -> _Draw | None:
    """F6 -- the Hard tier: one line item decides which years count, another supplies the answer.

    This is what the organisers call multi-hop dependent, and it is 21.3% of the paper: "among
    the years revenue grew, the one with the highest asset turnover". The dependency is the whole
    point, so the filter and the answer must read *different* line items. Filtering and ranking
    the same one is not multi-hop at all -- picking the largest of the members above their own
    median returns the same value as picking the largest outright, and a set built that way would
    look like it covered the tier while testing nothing.
    """
    paired = _shared_label_pairs(frames, rng)
    if paired is None:
        return None
    gate_rows, answer_rows = paired
    ordered = sorted(gate_rows)
    if len(ordered) < 3:
        return None

    def cell_of(rows: dict[str, pd.Series], name: str) -> CellExpr:
        row = rows[name]
        return CellExpr(
            variable=str(row["variable"]),
            row_index=int(row["row_index"]),
            column_index=int(row["column_index"]),
        )

    gates = tuple(cell_of(gate_rows, name) for name in ordered)
    answers = tuple(cell_of(answer_rows, name) for name in ordered)
    gate_values = [float(gate_rows[name]["base_value"]) for name in ordered]
    answer_values = [float(answer_rows[name]["base_value"]) for name in ordered]

    # The median leaves only one year above it for spans of three or four, and a lone survivor
    # makes the ranking vacuous -- so every sample that survived was a five-year one and two
    # thirds of the draws were wasted. One rank lower always leaves at least two.
    threshold = sorted(gate_values)[max(0, len(gate_values) // 2 - 1)]
    survivors = [index for index, value in enumerate(gate_values) if value > threshold]
    # One survivor makes the ranking vacuous, and the answer must actually depend on the filter,
    # otherwise the sample is a plain extremum wearing a condition.
    if len(survivors) < 2:
        return None
    picked = max(survivors, key=lambda index: answer_values[index])
    if picked == max(range(len(answer_values)), key=lambda index: answer_values[index]):
        return None

    gate_dimension = _cell_dimension(gate_rows[ordered[0]])
    if gate_dimension is None or any(
        _cell_dimension(gate_rows[name]) != gate_dimension for name in ordered
    ):
        return None
    condition = Condition(
        left=gates,
        comparator=">",
        # A threshold compared against VND cells is itself in VND. Left dimensionless it made
        # every Hard sample incomparable with its own filter.
        right=LiteralExpr(value=threshold, dimension=cast(Dimension, gate_dimension)),
    )
    expression: ScalarExpr = SelectExpr(
        operator="argmax", members=answers, conditions=(condition,), keys=answers
    )
    anchor = answer_rows[ordered[-1]]
    chosen = _askable_unit(anchor, rng)
    if chosen is None:
        return None
    unit_text, unit_name, divisor = chosen
    return _Draw(
        expression,
        unit_name,
        unit_text,
        divisor,
        str(anchor["row_label"]),
        str(anchor["column_label"]),
        # Survivors are exactly the years whose gate value beats a threshold taken from the gate
        # series itself, and every non-survivor sits at or below it -- so "the N years where this
        # item was highest" describes the same set exactly, and says it the way a question would.
        condition={
            "row_label": str(gate_rows[ordered[0]]["row_label"]),
            "top_n": len(survivors),
            "of_years": len(ordered),
            "comparator": ">",
            "threshold": threshold,
        },
    )


def _label_index(manifest: pd.DataFrame, key: tuple[str, str]) -> dict[int, dict[str, list[str]]]:
    """Which tables in each year carry which line item, read straight off the manifest.

    Matching multi-year draws by section heading throws most of them away: headings carry note
    numbers that move between years and "(tiếp theo)" continuations, so 31 of 50 sampled draws
    failed to assemble a year set at all. What the families actually need is a line item present
    in every year, and the manifest already lists every table's row labels, so the years can be
    matched on the thing being asked for instead of on the heading above it.
    """
    ticker, scope = key
    subset = manifest[(manifest["ticker"] == ticker) & (manifest["scope"] == scope)]
    index: dict[int, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for ref, year, labels in zip(
        subset["table_ref"], subset["report_year"], subset["row_labels"], strict=True
    ):
        for label in labels:
            folded = _normalise(str(label))
            if folded and _is_askable(str(label)):
                index[int(year)][folded].append(str(ref))
    return index


def _years_sharing_labels(
    index: dict[int, dict[str, list[str]]],
    years: list[int],
    rng: random.Random,
    *,
    minimum: int = 1,
) -> list[str] | None:
    """One table per year, all carrying the same `minimum` line items.

    The Hard family needs two: one line item to filter on and a different one to answer with.
    Assembling the year set on a single shared label and hoping a second turned up left it
    hunting for a pair it had never asked for -- 78% of its draws died there, and each had
    already paid to load and parse every table in the set.
    """
    if not years or any(year not in index for year in years):
        return None
    common = set(index[years[0]])
    for year in years[1:]:
        common &= set(index[year])
        if len(common) < minimum:
            return None
    labels = rng.sample(sorted(common), minimum)
    refs: list[str] = []
    for year in years:
        # One table per year has to carry all of them: the frames are what the sampler compares,
        # and a label sitting in a table nobody loaded is not shared with anything.
        carriers = set(index[year][labels[0]])
        for label in labels[1:]:
            carriers &= set(index[year][label])
        if not carriers:
            return None
        ordered = sorted(carriers)
        refs.append(ordered[rng.randrange(len(ordered))])
    return refs


def _frames_for(
    store: TableStore, refs: list[str]
) -> tuple[dict[str, pd.DataFrame], list[ManifestRecord]] | None:
    frames: dict[str, pd.DataFrame] = {}
    records: list[ManifestRecord] = []
    for position, ref in enumerate(refs, start=1):
        try:
            record, parsed = store.load(ref)
        except Exception:  # noqa: BLE001 - an unreadable table is skipped, not fatal
            return None
        frame = parsed_table_to_long_frame(record, parsed)
        # A table that parses to no rows yields an empty frame, and an empty frame carries no
        # columns at all -- every later `frame["base_value"]` then raises KeyError rather than
        # returning nothing. Reject it here so the failure stays local to one draw.
        if frame.empty:
            return None
        variable = f"df{position}"
        frame = frame.assign(variable=variable)
        frames[variable] = frame
        records.append(record)
    return frames, records


def _write(output: Path, kept: list[Sample]) -> None:
    """Persist what has been drawn so far, replacing the file atomically.

    A pool of several thousand takes hours, and holding all of it in memory until the last line
    means a run that dies at the two-hour mark leaves nothing behind. Writing through at
    intervals costs a fraction of a second and turns a total loss into a partial one.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_suffix(output.suffix + ".partial")
    with staging.open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(kept, start=1):
            handle.write(json.dumps({"id": index, **asdict(sample)}, ensure_ascii=False) + "\n")
    staging.replace(output)


def _usable_manifest(manifest: Path, min_rows: int, min_cols: int) -> pd.DataFrame:
    frame = pd.read_parquet(manifest)
    # A table whose unit was never resolved produces a number in an unknown scale, and a label
    # with the wrong scale is worse than no label at all.
    return frame[
        (frame["unit"] != "UNKNOWN")
        & (frame["n_rows"] >= min_rows)
        & (frame["n_cols"] >= min_cols)
        & frame["scope"].isin(["consolidated", "separate"])
    ].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data/processed/table_manifest.parquet"
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/ViFinQA")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1_000, help="How many samples to keep")
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--min-rows", type=int, default=3)
    parser.add_argument("--min-cols", type=int, default=2)
    parser.add_argument(
        "--tickers",
        help="Comma-separated tickers to draw from, so a split stays disjoint by company",
    )
    parser.add_argument(
        "--split",
        choices=["dev", "train"],
        # Naming the role beats pasting eighty tickers, and pasting eighty tickers beats what
        # actually happened: a dev set generated with neither, drawing from all hundred, which
        # belongs to neither side of the split and cannot be used by either.
        help="Draw only from this side of --ticker-split, so train and dev stay disjoint",
    )
    parser.add_argument(
        "--ticker-split",
        type=Path,
        default=ROOT / "outputs/synthetic/ticker_split.json",
    )
    parser.add_argument(
        "--attempts-per-sample",
        type=int,
        default=12,
        help="Give up on a drawn table after this many failed program shapes",
    )
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be at least 1")

    rng = random.Random(args.seed)
    manifest = _usable_manifest(args.manifest, args.min_rows, args.min_cols)
    if args.split and args.tickers:
        parser.error("--split and --tickers both choose the companies; pass one")
    split_used: list[str] | None = None
    if args.split:
        split = json.loads(args.ticker_split.read_text(encoding="utf-8"))
        split_used = sorted(str(ticker).upper() for ticker in split[args.split])
    elif args.tickers:
        split_used = sorted(
            item.strip().upper() for item in args.tickers.split(",") if item.strip()
        )
    if split_used is not None:
        manifest = manifest[manifest["ticker"].isin(set(split_used))].reset_index(drop=True)
    if manifest.empty:
        parser.error("no usable tables after filtering")

    families = FAMILY_WEIGHTS
    by_document: dict[str, list[str]] = defaultdict(list)
    for ref, doc in zip(manifest["table_ref"], manifest["doc_id"], strict=True):
        by_document[doc].append(ref)
    by_ticker_year: dict[tuple[str, str], dict[int, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for ref, ticker, year, scope in zip(
        manifest["table_ref"],
        manifest["ticker"],
        manifest["report_year"],
        manifest["scope"],
        strict=True,
    ):
        by_ticker_year[(ticker, scope)][int(year)].append(ref)

    label_indexes: dict[tuple[str, str], dict[int, dict[str, list[str]]]] = {}
    documents = sorted(by_document)
    multi_year = [key for key, years in by_ticker_year.items() if len(years) >= 2]
    # One store for the whole run: rebuilding it per draw re-reads the manifest every time.
    records = [ManifestRecord.from_dict(row) for row in manifest.to_dict("records")]
    store = TableStore(args.data_root, records)

    # Draw against a quota rather than a weighted coin. Weighting the draw skews the kept set
    # towards whichever family succeeds most often -- a first run asking for 46% lookups kept
    # 77% of them -- and the point of the set is to mirror the real question mix, not the
    # sampler's own hit rate.
    quota = {name: max(1, round(args.count * weight)) for name, weight in families}
    counts: dict[str, int] = defaultdict(int)
    kept: list[Sample] = []
    attempts = 0
    limit = args.count * args.attempts_per_sample
    while len(kept) < args.count and attempts < limit:
        attempts += 1
        short = [name for name, _ in families if counts[name] < quota[name]]
        if not short:
            break
        family = short[rng.randrange(len(short))]
        if family in {"lookup", "ratio"}:
            document = documents[rng.randrange(len(documents))]
            refs = [by_document[document][rng.randrange(len(by_document[document]))]]
        else:
            if not multi_year:
                continue
            key = multi_year[rng.randrange(len(multi_year))]
            years = sorted(by_ticker_year[key])
            span = 2 if family == "change" else min(len(years), rng.randint(3, 5))
            if len(years) < span:
                continue
            start = rng.randrange(len(years) - span + 1)
            chosen_years = years[start : start + span]
            # Years are matched on a line item they all carry, not on the heading above it.
            if key not in label_indexes:
                label_indexes[key] = _label_index(manifest, key)
            found_refs = _years_sharing_labels(
                label_indexes[key],
                chosen_years,
                rng,
                minimum=2 if family == "conditional" else 1,
            )
            if found_refs is None:
                continue
            refs = found_refs
        loaded = _frames_for(store, refs)
        if loaded is None:
            continue
        frames, records = loaded
        if family == "lookup":
            drawn: _Draw | None = _sample_lookup(frames["df1"], records[0], rng)
        elif family == "ratio":
            drawn = _sample_ratio(frames["df1"], records[0], rng)
        elif family == "change":
            drawn = _sample_change(frames, rng)
        elif family == "extremum":
            drawn = _sample_extremum(frames, rng)
        else:
            drawn = _sample_conditional(frames, rng)
        if drawn is None:
            continue
        expression, unit_name, unit_text, divisor, row_label, column_label, condition = drawn
        executed = _execute(expression, frames, unit_name, divisor)
        if executed is None:
            continue
        query, answer = executed
        first = records[0]
        counts[family] += 1
        if len(kept) and len(kept) % 250 == 0:
            _write(args.output, kept)
            print(
                f"  {len(kept)}/{args.count} kept after {attempts} attempts",
                flush=True,
            )
        kept.append(
            Sample(
                family=family,
                table_refs=tuple(refs),
                doc_ids=tuple(str(record.doc_id) for record in records),
                ticker=str(first.ticker),
                report_years=tuple(int(record.report_year) for record in records),
                scope=str(first.scope),
                row_label=row_label,
                column_label=column_label,
                section_title=first.section_title,
                target_unit=unit_name,
                target_unit_text=unit_text,
                target_divisor=divisor,
                answer=answer,
                pandas_query=query,
                program=expression_to_dict(expression),
                condition=condition,
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write(args.output, kept)

    args.output.with_suffix(".split.json").write_text(
        json.dumps(
            {
                "split": args.split,
                "tickers": split_used,
                "seed": args.seed,
                "count": len(kept),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"kept {len(kept)} samples from {attempts} attempts -> {args.output}")
    if args.split:
        drawn_from = f"the {args.split} split, {len(split_used or [])} tickers"
    elif split_used:
        drawn_from = f"{len(split_used)} named tickers"
    else:
        drawn_from = "every ticker -- this set belongs to neither side of the split"
    print(f"  drawn from: {drawn_from}")
    print("  by family: " + ", ".join(f"{name} {counts[name]}" for name in sorted(counts)))
    if len(kept) < args.count:
        print(f"  short of --count {args.count}: raise --attempts-per-sample or widen --tickers")


if __name__ == "__main__":
    main()
