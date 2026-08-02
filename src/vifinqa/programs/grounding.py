from __future__ import annotations

import math
from collections.abc import Mapping

import pandas as pd

from vifinqa.parsing.normalize import ascii_words
from vifinqa.programs.dimensions import infer_dimension
from vifinqa.programs.ir import (
    AggregateExpr,
    ArgExtremumExpr,
    BinaryExpr,
    CellExpr,
    CountIfExpr,
    Dimension,
    LiteralExpr,
    ScalarExpr,
)

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
_TARGET_DIMENSIONS: dict[str, Dimension] = {
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
    raise TypeError(f"Unsupported expression: {type(expression).__name__}")


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
    raise TypeError(f"Unsupported expression: {type(expression).__name__}")


def _validate_cells(expression: ScalarExpr, frames: Mapping[str, pd.DataFrame]) -> None:
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
        if actual_dimension == "UNKNOWN" and cell.dimension in {"VND", "USD", "SHARES"}:
            raise ValueError("Currency/share dimension requires explicit source-unit lineage")
        value = matches.iloc[0][cell.value_column]
        if pd.isna(value):
            raise ValueError(
                f"Selected cell has no numeric value: {cell.variable} "
                f"r{cell.row_index}/c{cell.column_index}"
            )


def validate_query_coverage(
    expression: ScalarExpr,
    *,
    frames: Mapping[str, pd.DataFrame],
    required_tickers: list[str],
    required_years: list[int],
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
    if missing_tickers:
        raise ValueError(f"Program does not cover required tickers: {missing_tickers}")

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
    if missing_years:
        raise ValueError(f"Program does not cover required years: {missing_years}")


def prepare_program(
    expression: ScalarExpr,
    *,
    selected_variables: list[str],
    frames: Mapping[str, pd.DataFrame],
    target_unit: str,
    target_divisor: float,
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
    _validate_cells(expression, frames)
    inferred = infer_dimension(expression)
    try:
        expected = _TARGET_DIMENSIONS[target_unit]
    except KeyError as exc:
        raise ValueError(f"Unsupported target unit: {target_unit}") from exc
    compatible = inferred == expected or (expected == "PERCENT" and inferred == "RATIO")
    if not compatible:
        raise ValueError(f"Program dimension {inferred} is incompatible with target {expected}")
    if not math.isfinite(target_divisor) or target_divisor <= 0:
        raise ValueError("target_divisor must be positive and finite")
    if target_divisor == 1.0:
        return expression, inferred
    return BinaryExpr("/", expression, LiteralExpr(target_divisor)), inferred
