from __future__ import annotations

from vifinqa.programs.ir import (
    AggregateExpr,
    ArgExtremumExpr,
    BinaryExpr,
    CellExpr,
    CountIfExpr,
    Dimension,
    LiteralExpr,
    ScalarExpr,
    SelectExpr,
)


def _compatible(left: Dimension, right: Dimension) -> Dimension:
    if left == "UNKNOWN":
        return right
    if right == "UNKNOWN":
        return left
    if left != right:
        raise ValueError(f"Incompatible dimensions: {left} and {right}")
    return left


def infer_dimension(expression: ScalarExpr) -> Dimension:
    if isinstance(expression, CellExpr):
        return expression.dimension
    if isinstance(expression, LiteralExpr):
        return expression.dimension
    if isinstance(expression, BinaryExpr):
        left = infer_dimension(expression.left)
        right = infer_dimension(expression.right)
        if expression.operator in {"+", "-"}:
            return _compatible(left, right)
        if expression.operator == "*":
            if left == "DIMENSIONLESS":
                return right
            if right == "DIMENSIONLESS":
                return left
            if "UNKNOWN" in {left, right}:
                return "UNKNOWN"
            raise ValueError(f"Unsupported dimensional multiplication: {left} * {right}")
        if right == "DIMENSIONLESS":
            return left
        if left == right and left != "UNKNOWN":
            return "RATIO"
        if "UNKNOWN" in {left, right}:
            return "UNKNOWN"
        raise ValueError(f"Unsupported dimensional division: {left} / {right}")
    if isinstance(expression, AggregateExpr):
        if not expression.operands:
            raise ValueError("Aggregate operands must not be empty")
        dimension = infer_dimension(expression.operands[0])
        for operand in expression.operands[1:]:
            dimension = _compatible(dimension, infer_dimension(operand))
        return dimension
    if isinstance(expression, CountIfExpr):
        threshold = infer_dimension(expression.threshold)
        if not expression.operands:
            raise ValueError("CountIf operands must not be empty")
        for operand in expression.operands:
            _compatible(infer_dimension(operand), threshold)
        return "COUNT"
    if isinstance(expression, SelectExpr):
        if not expression.members:
            raise ValueError("Select members must not be empty")
        for condition in expression.conditions:
            comparand = condition.right
            bounds = (
                comparand if isinstance(comparand, tuple) else (comparand,) * len(condition.left)
            )
            for operand, bound in zip(condition.left, bounds, strict=True):
                _compatible(infer_dimension(operand), infer_dimension(bound))
        if expression.operator == "count":
            return "COUNT"
        dimension = infer_dimension(expression.members[0])
        for member in expression.members[1:]:
            dimension = _compatible(dimension, infer_dimension(member))
        if expression.keys:
            key_dimension = infer_dimension(expression.keys[0])
            for key in expression.keys[1:]:
                key_dimension = _compatible(key_dimension, infer_dimension(key))
        return dimension
    if isinstance(expression, ArgExtremumExpr):
        if not expression.keys or len(expression.keys) != len(expression.values):
            raise ValueError("ArgExtremum keys and values must have the same non-zero length")
        key_dimension = infer_dimension(expression.keys[0])
        for key in expression.keys[1:]:
            key_dimension = _compatible(key_dimension, infer_dimension(key))
        value_dimension = infer_dimension(expression.values[0])
        for value in expression.values[1:]:
            value_dimension = _compatible(value_dimension, infer_dimension(value))
        return value_dimension
    raise TypeError(f"Unsupported expression: {type(expression).__name__}")
