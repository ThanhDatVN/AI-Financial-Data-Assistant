from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace

import pandas as pd

from vifinqa.parsing.normalize import ascii_words
from vifinqa.programs.dimensions import infer_dimension
from vifinqa.programs.ir import (
    AggregateExpr,
    ArgExtremumExpr,
    BinaryExpr,
    CellExpr,
    Condition,
    CountIfExpr,
    Dimension,
    LiteralExpr,
    ScalarExpr,
    SelectExpr,
)


class SoftGroundingError(ValueError):
    """A program this layer doubts, rather than one it knows will not run.

    The distinction decides where a question goes when the model runs out of attempts. A cell
    with no value would crash and a cancelled sum answers zero, so those must stay refusals. But
    a missing year or an undeclared unit only means the program disagrees with what we expected
    of it, and the alternative -- a keyword-matched cell from the same tables -- meets none of
    these standards either. Refusing is not the safe option there, only the tidier-looking one.
    """


_UNIT_DIMENSIONS: dict[str, Dimension] = {
    "VND": "VND",
    "THOUSAND_VND": "VND",
    "MILLION_VND": "VND",
    "BILLION_VND": "VND",
    "USD": "USD",
    "MILLION_USD": "USD",
    "PERCENT": "PERCENT",
    "SHARES": "SHARES",
}
TARGET_DIMENSIONS: dict[str, Dimension] = {
    "VND": "VND",
    "THOUSAND_VND": "VND",
    "MILLION_VND": "VND",
    "BILLION_VND": "VND",
    "HUNDRED_BILLION_VND": "VND",
    "TRILLION_VND": "VND",
    "MILLION_USD": "USD",
    "SHARES": "SHARES",
    "MILLION_SHARES": "SHARES",
    "PERCENT": "PERCENT",
    "RATIO": "RATIO",
    "COUNT": "COUNT",
    "YEAR": "YEAR",
}


def referenced_variables(expression: ScalarExpr) -> set[str]:
    if isinstance(expression, CellExpr):
        return {expression.variable}
    if isinstance(expression, LiteralExpr):
        return set()
    if isinstance(expression, BinaryExpr):
        return referenced_variables(expression.left) | referenced_variables(expression.right)
    if isinstance(expression, AggregateExpr | CountIfExpr):
        variables = set().union(*(referenced_variables(item) for item in expression.operands))
        if isinstance(expression, CountIfExpr):
            variables |= referenced_variables(expression.threshold)
        return variables
    if isinstance(expression, ArgExtremumExpr):
        return set().union(
            *(referenced_variables(item) for item in (*expression.keys, *expression.values))
        )
    if isinstance(expression, SelectExpr):
        return set().union(*(referenced_variables(item) for item in _select_parts(expression)))
    raise TypeError(f"Unsupported expression: {type(expression).__name__}")


def _select_parts(expression: SelectExpr) -> tuple[ScalarExpr, ...]:
    """Every sub-expression a selection reads: members, ranking keys and both sides of each
    condition."""
    parts: list[ScalarExpr] = [*expression.members, *(expression.keys or ())]
    for condition in expression.conditions:
        parts.extend(condition.left)
        right = condition.right
        parts.extend(right if isinstance(right, tuple) else (right,))
    return tuple(parts)


def cells_in_program(expression: ScalarExpr) -> tuple[CellExpr, ...]:
    """Every cell the program reads, in evaluation order."""
    return _cells(expression)


def _cells(expression: ScalarExpr) -> tuple[CellExpr, ...]:
    if isinstance(expression, CellExpr):
        return (expression,)
    if isinstance(expression, LiteralExpr):
        return ()
    if isinstance(expression, BinaryExpr):
        return (*_cells(expression.left), *_cells(expression.right))
    if isinstance(expression, AggregateExpr):
        return tuple(cell for operand in expression.operands for cell in _cells(operand))
    if isinstance(expression, CountIfExpr):
        return (
            *(cell for operand in expression.operands for cell in _cells(operand)),
            *_cells(expression.threshold),
        )
    if isinstance(expression, ArgExtremumExpr):
        return tuple(
            cell for item in (*expression.keys, *expression.values) for cell in _cells(item)
        )
    if isinstance(expression, SelectExpr):
        return tuple(cell for item in _select_parts(expression) for cell in _cells(item))
    raise TypeError(f"Unsupported expression: {type(expression).__name__}")


def _cell_source_unit(cell: CellExpr, frames: Mapping[str, pd.DataFrame]) -> str | None:
    frame = frames.get(cell.variable)
    if frame is None or "source_unit" not in frame.columns:
        return None
    matches = frame.loc[
        (frame["row_index"] == cell.row_index) & (frame["column_index"] == cell.column_index)
    ]
    if len(matches) != 1:
        return None
    return str(matches.iloc[0]["source_unit"])


def normalize_cells(expression: ScalarExpr, frames: Mapping[str, pd.DataFrame]) -> ScalarExpr:
    """Settle each cell's unit and column from its source lineage.

    Both follow from the source unit with no judgement left over: a scaled currency or share
    figure only means anything through `base_value`, a percentage only through
    `numeric_value`, and the dimension is whatever the unit says it is. A model that labels
    a million-VND cell as a percentage is not proposing a different reading of the evidence,
    it is guessing at a fact the evidence already fixes, so correct it instead of spending
    the question's remaining attempts on the same mistake.

    Cells whose source unit is unknown keep the claimed dimension, and `_validate_cells`
    still refuses a currency or share claim that has no lineage behind it.
    """
    if isinstance(expression, CellExpr):
        dimension = _UNIT_DIMENSIONS.get(_cell_source_unit(expression, frames) or "", "UNKNOWN")
        if dimension in {"VND", "USD", "SHARES"}:
            return replace(expression, value_column="base_value", dimension=dimension)
        if dimension == "PERCENT":
            return replace(expression, value_column="numeric_value", dimension=dimension)
        return expression
    if isinstance(expression, LiteralExpr):
        return expression
    if isinstance(expression, BinaryExpr):
        return replace(
            expression,
            left=normalize_cells(expression.left, frames),
            right=normalize_cells(expression.right, frames),
        )
    if isinstance(expression, AggregateExpr):
        return replace(
            expression,
            operands=tuple(normalize_cells(item, frames) for item in expression.operands),
        )
    if isinstance(expression, CountIfExpr):
        return replace(
            expression,
            operands=tuple(normalize_cells(item, frames) for item in expression.operands),
            threshold=normalize_cells(expression.threshold, frames),
        )
    if isinstance(expression, ArgExtremumExpr):
        return replace(
            expression,
            keys=tuple(normalize_cells(item, frames) for item in expression.keys),
            values=tuple(normalize_cells(item, frames) for item in expression.values),
        )
    if isinstance(expression, SelectExpr):
        conditions = tuple(
            Condition(
                left=tuple(normalize_cells(item, frames) for item in condition.left),
                comparator=condition.comparator,
                right=(
                    tuple(normalize_cells(item, frames) for item in condition.right)
                    if isinstance(condition.right, tuple)
                    else normalize_cells(condition.right, frames)
                ),
            )
            for condition in expression.conditions
        )
        return replace(
            expression,
            members=tuple(normalize_cells(item, frames) for item in expression.members),
            conditions=conditions,
            keys=(
                None
                if expression.keys is None
                else tuple(normalize_cells(item, frames) for item in expression.keys)
            ),
        )
    raise TypeError(f"Unsupported expression: {type(expression).__name__}")


def _validate_cells(
    expression: ScalarExpr, frames: Mapping[str, pd.DataFrame], *, lenient: bool = False
) -> None:
    for cell in _cells(expression):
        if cell.variable not in frames:
            raise ValueError(f"Program references unselected variable: {cell.variable}")
        frame = frames[cell.variable]
        required_columns = {"row_index", "column_index", "source_unit", cell.value_column}
        missing = required_columns - set(frame.columns)
        if missing:
            raise ValueError(f"Evidence {cell.variable} is missing columns: {sorted(missing)}")
        matches = frame.loc[
            (frame["row_index"] == cell.row_index) & (frame["column_index"] == cell.column_index)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Cell coordinate must resolve exactly once: {cell.variable} "
                f"r{cell.row_index}/c{cell.column_index}, matches={len(matches)}"
            )
        source_unit = str(matches.iloc[0]["source_unit"])
        actual_dimension = _UNIT_DIMENSIONS.get(source_unit, "UNKNOWN")
        if actual_dimension != "UNKNOWN" and cell.dimension != actual_dimension:
            raise ValueError(
                f"Cell dimension mismatch at {cell.variable} "
                f"r{cell.row_index}/c{cell.column_index}: "
                f"source_unit={source_unit}, claimed={cell.dimension}"
            )
        if actual_dimension in {"VND", "USD", "SHARES"} and cell.value_column != "base_value":
            raise ValueError("Scaled currency/share cells must use base_value")
        if actual_dimension == "PERCENT" and cell.value_column != "numeric_value":
            raise ValueError("Percentage cells must use numeric_value")
        if (
            not lenient
            and actual_dimension == "UNKNOWN"
            and cell.dimension in {"VND", "USD", "SHARES"}
        ):
            # 52 questions in the full run ended here, and every one of them was then answered
            # by the fallback -- which reads a keyword-matched cell out of the same undeclared
            # table, taking the identical scale risk with a worse choice of cell. Refusing the
            # model's program does not avoid the risk, it only discards the reasoning. So the
            # check stays strict while the model can still be told to fix it, and gives way on
            # the last attempt rather than handing the question to a guess.
            raise SoftGroundingError(
                "Currency/share dimension requires explicit source-unit lineage"
            )
        value = matches.iloc[0][cell.value_column]
        if pd.isna(value):
            # The retry loop only sees this message, so it has to say where the numbers are.
            # Without that the model re-picks a neighbouring blank and burns every attempt.
            populated = frame.loc[
                (frame["row_index"] == cell.row_index) & frame["numeric_value"].notna(),
                "column_index",
            ]
            available = ", ".join(f"c{int(column)}" for column in sorted(populated.unique()))
            raise ValueError(
                f"Selected cell has no numeric value: {cell.variable} "
                f"r{cell.row_index}/c{cell.column_index}. "
                + (
                    f"Row r{cell.row_index} holds numbers at {available}."
                    if available
                    else f"Row r{cell.row_index} holds no number at all; choose another row."
                )
            )


def validate_answer_plausibility(answer: float, expression: ScalarExpr) -> None:
    """Refuse a multi-cell program that collapses to exactly zero.

    A financial statement repeats the same figure across schedules, once with the opposite
    sign where it is backed out as an adjustment. Combining those restatements cancels them
    to a clean zero that satisfies every unit, coverage and execution check, so nothing else
    in the pipeline can tell that answer apart from a real one.
    """
    if answer == 0.0 and len(_cells(expression)) > 1:
        raise ValueError(
            "Multi-cell program evaluated to exactly zero, which usually means the same "
            "figure was combined with its sign-flipped restatement. Use only the cell that "
            "reports the requested figure."
        )


def validate_query_coverage(
    expression: ScalarExpr,
    *,
    frames: Mapping[str, pd.DataFrame],
    required_tickers: list[str],
    required_years: list[int],
    lenient: bool = False,
) -> None:
    covered_tickers: set[str] = set()
    cell_evidence: list[tuple[int, str]] = []
    for cell in _cells(expression):
        frame = frames[cell.variable]
        for metadata_column in ("ticker", "report_year", "column_label"):
            if metadata_column not in frame.columns:
                raise ValueError(
                    f"Evidence {cell.variable} is missing coverage column {metadata_column}"
                )
        matches = frame.loc[
            (frame["row_index"] == cell.row_index) & (frame["column_index"] == cell.column_index)
        ]
        row = matches.iloc[0]
        covered_tickers.add(str(row["ticker"]))
        cell_evidence.append((int(row["report_year"]), ascii_words(str(row["column_label"]))))

    missing_tickers = sorted(set(required_tickers) - covered_tickers)
    if missing_tickers and not lenient:
        raise SoftGroundingError(f"Program does not cover required tickers: {missing_tickers}")

    prior_markers = ("dau nam", "dau ky", "nam truoc", "ky truoc", "so dau")
    missing_years: list[int] = []
    for year in sorted(set(required_years)):
        covered = any(report_year == year for report_year, _ in cell_evidence)
        if not covered:
            covered = any(
                report_year == year + 1
                and (str(year) in column_label or any(x in column_label for x in prior_markers))
                for report_year, column_label in cell_evidence
            )
        if not covered:
            missing_years.append(year)
    if missing_years and not lenient:
        raise SoftGroundingError(f"Program does not cover required years: {missing_years}")


def snap_cells(expression: ScalarExpr, frames: Mapping[str, pd.DataFrame]) -> ScalarExpr:
    """Move a cell that holds no number onto its row's only numeric column.

    Last-resort only, for the same reason the other relaxations are: the alternative is a
    keyword-matched cell from a table nobody chose. 13 of the 21 failures on the widest questions
    were coordinate errors, and in some of them the row the model named holds exactly one number
    -- there is no guessing left to do, only a column index to correct.

    Ambiguity is left alone. A row with three populated columns needs a decision this function has
    no basis to make, and inventing one would be worse than refusing.
    """
    if isinstance(expression, CellExpr):
        frame = frames.get(expression.variable)
        if frame is None or "numeric_value" not in frame.columns:
            return expression
        here = frame.loc[
            (frame["row_index"] == expression.row_index)
            & (frame["column_index"] == expression.column_index)
        ]
        if len(here) == 1 and not pd.isna(here.iloc[0][expression.value_column]):
            return expression
        populated = frame.loc[
            (frame["row_index"] == expression.row_index) & frame["numeric_value"].notna(),
            "column_index",
        ].unique()
        if len(populated) != 1:
            return expression
        return replace(expression, column_index=int(populated[0]))
    if isinstance(expression, LiteralExpr):
        return expression
    if isinstance(expression, BinaryExpr):
        return replace(
            expression,
            left=snap_cells(expression.left, frames),
            right=snap_cells(expression.right, frames),
        )
    if isinstance(expression, AggregateExpr):
        return replace(
            expression, operands=tuple(snap_cells(item, frames) for item in expression.operands)
        )
    if isinstance(expression, CountIfExpr):
        return replace(
            expression,
            operands=tuple(snap_cells(item, frames) for item in expression.operands),
            threshold=snap_cells(expression.threshold, frames),
        )
    if isinstance(expression, ArgExtremumExpr):
        return replace(
            expression,
            keys=tuple(snap_cells(item, frames) for item in expression.keys),
            values=tuple(snap_cells(item, frames) for item in expression.values),
        )
    if isinstance(expression, SelectExpr):
        return replace(
            expression,
            members=tuple(snap_cells(item, frames) for item in expression.members),
            keys=None
            if expression.keys is None
            else tuple(snap_cells(item, frames) for item in expression.keys),
            conditions=tuple(
                replace(
                    condition,
                    left=tuple(snap_cells(item, frames) for item in condition.left),
                    right=tuple(snap_cells(item, frames) for item in condition.right)
                    if isinstance(condition.right, tuple)
                    else snap_cells(condition.right, frames),
                )
                for condition in expression.conditions
            ),
        )
    return expression


def prepare_program(
    expression: ScalarExpr,
    *,
    selected_variables: list[str],
    frames: Mapping[str, pd.DataFrame],
    target_unit: str,
    target_divisor: float,
    lenient: bool = False,
) -> tuple[ScalarExpr, Dimension]:
    if not selected_variables or len(selected_variables) != len(set(selected_variables)):
        raise ValueError("selected_variables must be non-empty and unique")
    referenced = referenced_variables(expression)
    if set(selected_variables) != referenced:
        raise ValueError(
            f"selected_variables must equal referenced variables: "
            f"selected={sorted(selected_variables)}, referenced={sorted(referenced)}"
        )
    if set(frames) != referenced:
        raise ValueError("Loaded frames must exactly match referenced variables")
    expression = normalize_cells(expression, frames)
    if lenient:
        expression = snap_cells(expression, frames)
    _validate_cells(expression, frames, lenient=lenient)
    inferred = infer_dimension(expression)
    try:
        expected = TARGET_DIMENSIONS[target_unit]
    except KeyError as exc:
        raise ValueError(f"Unsupported target unit: {target_unit}") from exc
    compatible = inferred == expected or (expected == "PERCENT" and inferred == "RATIO")
    if not compatible:
        remedy = ""
        if expected in {"PERCENT", "RATIO"} and inferred not in {"PERCENT", "RATIO"}:
            # Repeating "incompatible" three times taught the model nothing. A percentage is
            # built by dividing, never by relabelling an amount.
            remedy = (
                f" The question wants a proportion, so divide one {inferred} cell by another "
                "instead of returning an amount."
            )
            # 78 questions failed here in one run, and every one was a select node whose members
            # were raw cells. Saying "divide" is not enough when the division has to go in a
            # particular place: a select returns a member, so the members are what must become
            # ratios, and the figure being compared belongs in `keys`.
            if isinstance(expression, SelectExpr):
                remedy += (
                    " This is a select node, which answers with one of its `members`: make every "
                    "member that ratio, and move the figure you rank by into `keys`."
                )
        elif expected == "YEAR":
            # The same shape of mistake as the ratio one, at a different target, and it was left
            # without a remedy: 21 questions ended on a bare "incompatible with target YEAR" in
            # one run. The model is asked which year, finds the right extreme, and then reports
            # the amount it ranked by -- which is what a select node returns unless its members
            # are the years. It has never once emitted `arg_extremum` (0 of 519 clean programs),
            # so naming the node it does use is what makes the remedy actionable.
            remedy = (
                " The question asks WHICH YEAR, so the answer is a year, not an amount. In a "
                "select node the answer is one of its `members`, so make the members the year "
                "literals and put the amounts you rank them by into `keys`; `arg_extremum` "
                "expresses the same thing in one node."
            )
        raise ValueError(
            f"Program dimension {inferred} is incompatible with target {expected}.{remedy}"
        )
    if not math.isfinite(target_divisor) or target_divisor <= 0:
        raise ValueError("target_divisor must be positive and finite")
    if target_divisor == 1.0:
        return expression, inferred
    return BinaryExpr("/", expression, LiteralExpr(target_divisor)), inferred
