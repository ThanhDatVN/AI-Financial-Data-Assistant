from __future__ import annotations

import re

from vifinqa.programs.ir import (
    AggregateExpr,
    ArgExtremumExpr,
    BinaryExpr,
    CellExpr,
    CountIfExpr,
    LiteralExpr,
    ScalarExpr,
    SelectExpr,
)

_VARIABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class _Bindings:
    """Name a sub-expression once so a cohort can read it many times.

    A selection compares every member against every other, and each comparison repeats the
    membership test, which in turn may repeat a median over the whole cohort. Inlined, a
    nine-company question expands past a megabyte; named, it stays a few kilobytes.
    """

    def __init__(self) -> None:
        self.assignments: list[str] = []

    def bind(self, compiled: str) -> str:
        name = f"_v{len(self.assignments)}"
        self.assignments.append(f"({name} := {compiled})")
        return name

    def wrap(self, body: str) -> str:
        if not self.assignments:
            return body
        return "(" + ", ".join([*self.assignments, body]) + ")[-1]"


def compile_expression(expression: ScalarExpr) -> str:
    bindings = _Bindings()
    return bindings.wrap(_compile(expression, bindings))


def _compile_conditions(expression: SelectExpr, size: int, bindings: _Bindings) -> list[str]:
    """Render one membership test per member, true when every condition holds for it."""
    per_member: list[list[str]] = [[] for _ in range(size)]
    for condition in expression.conditions:
        if len(condition.left) != size:
            raise ValueError("Condition operands must align with the cohort members")
        if isinstance(condition.right, tuple):
            if len(condition.right) != size:
                raise ValueError("Per-member thresholds must align with the cohort members")
            thresholds = [_compile(item, bindings) for item in condition.right]
        else:
            thresholds = [bindings.bind(_compile(condition.right, bindings))] * size
        for index, operand in enumerate(condition.left):
            left = _compile(operand, bindings)
            per_member[index].append(f"({left} {condition.comparator} {thresholds[index]})")
    return [bindings.bind(" and ".join(tests)) if tests else "True" for tests in per_member]


def _stable_rank(values: list[str], index: int) -> str:
    """Rank one member among its cohort, breaking ties by position."""
    tests = []
    for other, value in enumerate(values):
        if other == index:
            continue
        if other < index:
            tests.append(f"(({value} < {values[index]}) or ({value} == {values[index]}))")
        else:
            tests.append(f"({value} < {values[index]})")
    return f"sum([{', '.join(tests)}])" if tests else "0"


def _select_by_rank(values: list[str], target_rank: int) -> str:
    selected = "float('nan')"
    for index in range(len(values) - 1, -1, -1):
        selected = (
            f"({values[index]} if ({_stable_rank(values, index)} == {target_rank}) "
            f"else {selected})"
        )
    return selected


def _compile_select(expression: SelectExpr, bindings: _Bindings) -> str:
    size = len(expression.members)
    if size == 0:
        raise ValueError("Select members must not be empty")
    if expression.operator in {"argmin", "argmax"}:
        if expression.keys is None or len(expression.keys) != size:
            raise ValueError("Ranked selection needs one key per member")
    elif expression.keys is not None:
        raise ValueError("Keys are only meaningful for argmin and argmax")

    members = [bindings.bind(_compile(member, bindings)) for member in expression.members]
    kept = _compile_conditions(expression, size, bindings)
    kept_count = f"sum([{', '.join(kept)}])"

    if expression.operator == "count":
        return kept_count
    if expression.operator == "sum":
        terms = [f"({member}) * ({test})" for member, test in zip(members, kept, strict=True)]
        return f"sum([{', '.join(terms)}])"
    if expression.operator == "mean":
        terms = [f"({member}) * ({test})" for member, test in zip(members, kept, strict=True)]
        return f"(sum([{', '.join(terms)}]) / {kept_count})"
    if expression.operator == "median":
        if expression.conditions:
            raise ValueError("Median over a filtered cohort is not supported")
        if size % 2:
            return _select_by_rank(members, (size - 1) // 2)
        lower = _select_by_rank(members, size // 2 - 1)
        upper = _select_by_rank(members, size // 2)
        return f"(({lower} + {upper}) / 2)"

    # min and max are the ranked selection with the member as its own key. Masking with a
    # sentinel would make the answer depend on argument order once NaN entered the list.
    keys = (
        members
        if expression.operator in {"min", "max"}
        else [bindings.bind(_compile(key, bindings)) for key in expression.keys or ()]
    )
    comparator = "<=" if expression.operator in {"argmin", "min"} else ">="
    selected = "float('nan')"
    for index in range(size - 1, -1, -1):
        rivals = [
            f"(({kept[other]} == False) or ({keys[index]} {comparator} {keys[other]}))"
            for other in range(size)
            if other != index
        ]
        wins = " and ".join(rivals) if rivals else "True"
        selected = f"({members[index]} if (({kept[index]}) and ({wins})) else {selected})"
    return selected


def _compile(expression: ScalarExpr, bindings: _Bindings) -> str:
    if isinstance(expression, LiteralExpr):
        return repr(float(expression.value))
    if isinstance(expression, CellExpr):
        if not _VARIABLE_RE.fullmatch(expression.variable):
            raise ValueError(f"Invalid evidence variable: {expression.variable!r}")
        if expression.row_index < 0 or expression.column_index < 0:
            raise ValueError("Cell coordinates must be non-negative")
        mask = (
            f"({expression.variable}['row_index'] == {expression.row_index}) & "
            f"({expression.variable}['column_index'] == {expression.column_index})"
        )
        return f"float({expression.variable}.loc[{mask}, '{expression.value_column}'].iloc[0])"
    if isinstance(expression, BinaryExpr):
        left = _compile(expression.left, bindings)
        right = _compile(expression.right, bindings)
        return f"({left} {expression.operator} {right})"
    if isinstance(expression, AggregateExpr):
        if not expression.operands:
            raise ValueError("Aggregate operands must not be empty")
        operands = [_compile(operand, bindings) for operand in expression.operands]
        rendered = f"[{', '.join(operands)}]"
        if expression.operator == "mean":
            return f"(sum({rendered}) / {len(operands)})"
        return f"{expression.operator}({rendered})"
    if isinstance(expression, CountIfExpr):
        if not expression.operands:
            raise ValueError("CountIf operands must not be empty")
        threshold = bindings.bind(_compile(expression.threshold, bindings))
        conditions = [
            f"({_compile(operand, bindings)} {expression.comparator} {threshold})"
            for operand in expression.operands
        ]
        return f"sum([{', '.join(conditions)}])"
    if isinstance(expression, SelectExpr):
        return _compile_select(expression, bindings)
    if isinstance(expression, ArgExtremumExpr):
        if not expression.keys or len(expression.keys) != len(expression.values):
            raise ValueError("ArgExtremum keys and values must have the same non-zero length")
        keys = [bindings.bind(_compile(key, bindings)) for key in expression.keys]
        values = [_compile(value, bindings) for value in expression.values]
        result = values[-1]
        comparator = "<=" if expression.mode == "argmin" else ">="
        for index in range(len(keys) - 2, -1, -1):
            comparisons = [
                f"({keys[index]} {comparator} {other})"
                for other_index, other in enumerate(keys)
                if other_index != index
            ]
            condition = " and ".join(comparisons)
            result = f"({values[index]} if ({condition}) else {result})"
        return result
    raise TypeError(f"Unsupported expression: {type(expression).__name__}")
