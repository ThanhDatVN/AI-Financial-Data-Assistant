from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

type Dimension = Literal[
    "VND",
    "USD",
    "PERCENT",
    "RATIO",
    "SHARES",
    "COUNT",
    "YEAR",
    "DIMENSIONLESS",
    "UNKNOWN",
]


@dataclass(frozen=True, slots=True)
class CellExpr:
    variable: str
    row_index: int
    column_index: int
    value_column: Literal["numeric_value", "base_value"] = "base_value"
    dimension: Dimension = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class LiteralExpr:
    value: float
    dimension: Dimension = "DIMENSIONLESS"


@dataclass(frozen=True, slots=True)
class BinaryExpr:
    operator: Literal["+", "-", "*", "/"]
    left: ScalarExpr
    right: ScalarExpr


@dataclass(frozen=True, slots=True)
class AggregateExpr:
    operator: Literal["sum", "mean", "min", "max"]
    operands: tuple[ScalarExpr, ...]


@dataclass(frozen=True, slots=True)
class CountIfExpr:
    operands: tuple[ScalarExpr, ...]
    comparator: Literal["<", "<=", ">", ">=", "==", "!="]
    threshold: ScalarExpr


@dataclass(frozen=True, slots=True)
class ArgExtremumExpr:
    mode: Literal["argmin", "argmax"]
    keys: tuple[ScalarExpr, ...]
    values: tuple[ScalarExpr, ...]


type ScalarExpr = (
    CellExpr | LiteralExpr | BinaryExpr | AggregateExpr | CountIfExpr | ArgExtremumExpr
)
