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


type Comparator = Literal["<", "<=", ">", ">=", "==", "!="]


@dataclass(frozen=True, slots=True)
class Condition:
    """One test applied member by member across a cohort.

    `left` holds one expression per member. `right` is either a single threshold shared by
    every member, or one expression per member for a pairwise comparison.
    """

    left: tuple[ScalarExpr, ...]
    comparator: Comparator
    right: ScalarExpr | tuple[ScalarExpr, ...]


@dataclass(frozen=True, slots=True)
class SelectExpr:
    """Aggregate over the members of a cohort that satisfy every condition.

    Counting a predicate was never the hard part; selecting by one was. A question that asks
    for the total revenue of the companies whose margin stayed positive, or for the leader
    within the half of a group below the median, needs the subset itself rather than its
    size. `median` falls out of the same node with no conditions attached.
    """

    operator: Literal["sum", "mean", "min", "max", "median", "count", "argmin", "argmax"]
    members: tuple[ScalarExpr, ...]
    conditions: tuple[Condition, ...] = ()
    keys: tuple[ScalarExpr, ...] | None = None


type ScalarExpr = (
    CellExpr | LiteralExpr | BinaryExpr | AggregateExpr | CountIfExpr | ArgExtremumExpr | SelectExpr
)
