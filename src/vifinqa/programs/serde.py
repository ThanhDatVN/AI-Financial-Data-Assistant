from __future__ import annotations

import copy
import math
import re
from typing import Literal, cast

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

DIMENSIONS: tuple[Dimension, ...] = (
    "VND",
    "USD",
    "PERCENT",
    "RATIO",
    "SHARES",
    "COUNT",
    "YEAR",
    "DIMENSIONLESS",
    "UNKNOWN",
)
_VARIABLE_RE = re.compile(r"^df[1-9][0-9]*$")

_EXPRESSION_REF: dict[str, object] = {"$ref": "#/$defs/expression"}
PROGRAM_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "selected_variables": {
            "type": "array",
            "items": {"type": "string", "pattern": "^df[1-9][0-9]*$"},
            "minItems": 1,
            "uniqueItems": True,
        },
        "program": _EXPRESSION_REF,
    },
    "required": ["selected_variables", "program"],
    "additionalProperties": False,
    "$defs": {
        "expression": {
            "oneOf": [
                {"$ref": "#/$defs/cell"},
                {"$ref": "#/$defs/literal"},
                {"$ref": "#/$defs/binary"},
                {"$ref": "#/$defs/aggregate"},
                {"$ref": "#/$defs/count_if"},
                {"$ref": "#/$defs/arg_extremum"},
            ]
        },
        "cell": {
            "type": "object",
            "properties": {
                "kind": {"const": "cell"},
                "variable": {"type": "string", "pattern": "^df[1-9][0-9]*$"},
                "row_index": {"type": "integer", "minimum": 0},
                "column_index": {"type": "integer", "minimum": 0},
                "value_column": {"enum": ["numeric_value", "base_value"]},
                "dimension": {"enum": list(DIMENSIONS)},
            },
            "required": [
                "kind",
                "variable",
                "row_index",
                "column_index",
                "value_column",
                "dimension",
            ],
            "additionalProperties": False,
        },
        "literal": {
            "type": "object",
            "properties": {
                "kind": {"const": "literal"},
                "value": {"type": "number"},
                "dimension": {"enum": list(DIMENSIONS)},
            },
            "required": ["kind", "value", "dimension"],
            "additionalProperties": False,
        },
        "binary": {
            "type": "object",
            "properties": {
                "kind": {"const": "binary"},
                "operator": {"enum": ["+", "-", "*", "/"]},
                "left": _EXPRESSION_REF,
                "right": _EXPRESSION_REF,
            },
            "required": ["kind", "operator", "left", "right"],
            "additionalProperties": False,
        },
        "aggregate": {
            "type": "object",
            "properties": {
                "kind": {"const": "aggregate"},
                "operator": {"enum": ["sum", "mean", "min", "max"]},
                "operands": {
                    "type": "array",
                    "items": _EXPRESSION_REF,
                    "minItems": 1,
                    "maxItems": 100,
                },
            },
            "required": ["kind", "operator", "operands"],
            "additionalProperties": False,
        },
        "count_if": {
            "type": "object",
            "properties": {
                "kind": {"const": "count_if"},
                "operands": {
                    "type": "array",
                    "items": _EXPRESSION_REF,
                    "minItems": 1,
                    "maxItems": 100,
                },
                "comparator": {"enum": ["<", "<=", ">", ">=", "==", "!="]},
                "threshold": _EXPRESSION_REF,
            },
            "required": ["kind", "operands", "comparator", "threshold"],
            "additionalProperties": False,
        },
        "arg_extremum": {
            "type": "object",
            "properties": {
                "kind": {"const": "arg_extremum"},
                "mode": {"enum": ["argmin", "argmax"]},
                "keys": {
                    "type": "array",
                    "items": _EXPRESSION_REF,
                    "minItems": 1,
                    "maxItems": 100,
                },
                "values": {
                    "type": "array",
                    "items": _EXPRESSION_REF,
                    "minItems": 1,
                    "maxItems": 100,
                },
            },
            "required": ["kind", "mode", "keys", "values"],
            "additionalProperties": False,
        },
    },
}

# vLLM 0.19.1 rejects the otherwise valid JSON Schema keyword
# ``uniqueItems`` when compiling a structured-output grammar. Preserve the
# canonical schema and send a backend-compatible copy to constrained decoding;
# the generation runner enforces uniqueness after decoding.
PROGRAM_GRAMMAR_SCHEMA: dict[str, object] = copy.deepcopy(PROGRAM_JSON_SCHEMA)
selected_variables_schema = cast(
    dict[str, object],
    cast(dict[str, object], PROGRAM_GRAMMAR_SCHEMA["properties"])["selected_variables"],
)
selected_variables_schema.pop("uniqueItems", None)


def _object(
    raw: object,
    *,
    required: set[str],
    depth: int,
) -> dict[str, object]:
    if depth > 20:
        raise ValueError("Program exceeds maximum nesting depth")
    if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
        raise TypeError("Every program node must be an object")
    node = cast(dict[str, object], raw)
    if set(node) != required:
        raise ValueError(
            f"Program node fields mismatch; expected={sorted(required)}, actual={sorted(node)}"
        )
    return node


def _enum(value: object, allowed: set[str], *, field: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"Invalid {field}: {value!r}")
    return value


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _items(value: object, *, field: str) -> list[object]:
    if not isinstance(value, list) or not value or len(value) > 100:
        raise ValueError(f"{field} must contain 1..100 nodes")
    return value


def expression_from_dict(raw: object, *, _depth: int = 0) -> ScalarExpr:
    if not isinstance(raw, dict):
        raise TypeError("Program node must be an object")
    kind = raw.get("kind")
    if kind == "cell":
        node = _object(
            raw,
            required={
                "kind",
                "variable",
                "row_index",
                "column_index",
                "value_column",
                "dimension",
            },
            depth=_depth,
        )
        variable = node["variable"]
        if not isinstance(variable, str) or not _VARIABLE_RE.fullmatch(variable):
            raise ValueError("variable must match dfN")
        value_column = _enum(
            node["value_column"], {"numeric_value", "base_value"}, field="value_column"
        )
        dimension = cast(Dimension, _enum(node["dimension"], set(DIMENSIONS), field="dimension"))
        return CellExpr(
            variable=variable,
            row_index=_integer(node["row_index"], field="row_index"),
            column_index=_integer(node["column_index"], field="column_index"),
            value_column=cast(Literal["numeric_value", "base_value"], value_column),
            dimension=dimension,
        )
    if kind == "literal":
        node = _object(
            raw,
            required={"kind", "value", "dimension"},
            depth=_depth,
        )
        return LiteralExpr(
            _number(node["value"], field="value"),
            cast(Dimension, _enum(node["dimension"], set(DIMENSIONS), field="dimension")),
        )
    if kind == "binary":
        node = _object(
            raw,
            required={"kind", "operator", "left", "right"},
            depth=_depth,
        )
        operator = _enum(node["operator"], {"+", "-", "*", "/"}, field="operator")
        return BinaryExpr(
            cast(Literal["+", "-", "*", "/"], operator),
            expression_from_dict(node["left"], _depth=_depth + 1),
            expression_from_dict(node["right"], _depth=_depth + 1),
        )
    if kind == "aggregate":
        node = _object(
            raw,
            required={"kind", "operator", "operands"},
            depth=_depth,
        )
        operator = _enum(node["operator"], {"sum", "mean", "min", "max"}, field="operator")
        return AggregateExpr(
            cast(Literal["sum", "mean", "min", "max"], operator),
            tuple(
                expression_from_dict(item, _depth=_depth + 1)
                for item in _items(node["operands"], field="operands")
            ),
        )
    if kind == "count_if":
        node = _object(
            raw,
            required={"kind", "operands", "comparator", "threshold"},
            depth=_depth,
        )
        comparator = _enum(
            node["comparator"], {"<", "<=", ">", ">=", "==", "!="}, field="comparator"
        )
        return CountIfExpr(
            tuple(
                expression_from_dict(item, _depth=_depth + 1)
                for item in _items(node["operands"], field="operands")
            ),
            cast(Literal["<", "<=", ">", ">=", "==", "!="], comparator),
            expression_from_dict(node["threshold"], _depth=_depth + 1),
        )
    if kind == "arg_extremum":
        node = _object(
            raw,
            required={"kind", "mode", "keys", "values"},
            depth=_depth,
        )
        mode = _enum(node["mode"], {"argmin", "argmax"}, field="mode")
        keys = tuple(
            expression_from_dict(item, _depth=_depth + 1)
            for item in _items(node["keys"], field="keys")
        )
        values = tuple(
            expression_from_dict(item, _depth=_depth + 1)
            for item in _items(node["values"], field="values")
        )
        if len(keys) != len(values):
            raise ValueError("arg_extremum keys and values must have equal length")
        return ArgExtremumExpr(cast(Literal["argmin", "argmax"], mode), keys, values)
    raise ValueError(f"Unknown program node kind: {kind!r}")
