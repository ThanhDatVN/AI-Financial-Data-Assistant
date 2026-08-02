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
)

_VARIABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def compile_expression(expression: ScalarExpr) -> str:
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
        return f"float({expression.variable}.loc[{mask}, " f"'{expression.value_column}'].iloc[0])"
    if isinstance(expression, BinaryExpr):
        left = compile_expression(expression.left)
        right = compile_expression(expression.right)
        return f"({left} {expression.operator} {right})"
    if isinstance(expression, AggregateExpr):
        if not expression.operands:
            raise ValueError("Aggregate operands must not be empty")
        operands = [compile_expression(operand) for operand in expression.operands]
        rendered = f"[{', '.join(operands)}]"
        if expression.operator == "mean":
            return f"(sum({rendered}) / {len(operands)})"
        return f"{expression.operator}({rendered})"
    if isinstance(expression, CountIfExpr):
        if not expression.operands:
            raise ValueError("CountIf operands must not be empty")
        threshold = compile_expression(expression.threshold)
        conditions = [
            f"({compile_expression(operand)} {expression.comparator} {threshold})"
            for operand in expression.operands
        ]
        return f"sum([{', '.join(conditions)}])"
    if isinstance(expression, ArgExtremumExpr):
        if not expression.keys or len(expression.keys) != len(expression.values):
            raise ValueError("ArgExtremum keys and values must have the same non-zero length")
        keys = [compile_expression(key) for key in expression.keys]
        values = [compile_expression(value) for value in expression.values]
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
