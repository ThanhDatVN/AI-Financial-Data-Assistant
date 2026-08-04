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
# Grounding derives both from the cell's source unit, so a program need not state them.
CELL_LINEAGE_FIELDS = frozenset({"value_column", "dimension"})

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
            "required": ["kind", "variable", "row_index", "column_index"],
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

# What the parser accepts. Programs are validated against these bounds after decoding.
PARSER_MAX_DEPTH = 20
PARSER_MAX_ITEMS = 100

# What the decoder is allowed to reach. Constrained decoding compiles the schema into a
# grammar up front, and that cost grows with every unrolled level and every bounded
# repetition: at the parser budget it becomes 103 definitions containing 80 arrays of 100
# items, which stalls XGrammar compilation before the first token is sampled.
#
# The width has to clear the widest question in the release rather than a comfortable round
# number. Routing fans out to `tickers x years`, which reaches 18 on question 442, so a
# cohort program there needs 18 operands per array. Depth has room to spare because real
# programs stay shallow: a ratio of sums is two levels and a growth rate is three.
MAX_ROUTE_FAN_OUT = 18  # measured across all 1,012 questions of the frozen retrieval
GRAMMAR_MAX_DEPTH = 6
GRAMMAR_MAX_ITEMS = 32


def _grammar_ref(name: str) -> dict[str, object]:
    return {"$ref": f"#/$defs/{name}"}


def _grammar_variant(value: object, *, depth: int) -> object:
    if isinstance(value, dict):
        if value == _EXPRESSION_REF:
            return _grammar_ref(f"expression_{depth}")
        return {
            str(key): (
                GRAMMAR_MAX_ITEMS
                if key == "maxItems" and item == PARSER_MAX_ITEMS
                else _grammar_variant(item, depth=depth)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_grammar_variant(item, depth=depth) for item in value]
    return copy.deepcopy(value)


def _build_program_grammar_schema() -> dict[str, object]:
    """Build an acyclic vLLM grammar small enough to compile before the first token."""

    schema = copy.deepcopy(PROGRAM_JSON_SCHEMA)
    properties = cast(dict[str, object], schema["properties"])
    selected_variables = cast(dict[str, object], properties["selected_variables"])
    # vLLM 0.19.1 rejects this valid JSON Schema validation keyword. The
    # generation runner enforces the same invariant after constrained decoding.
    selected_variables.pop("uniqueItems", None)

    canonical_defs = cast(dict[str, object], schema["$defs"])
    grammar_defs: dict[str, object] = {
        "cell": copy.deepcopy(canonical_defs["cell"]),
        "literal": copy.deepcopy(canonical_defs["literal"]),
    }
    recursive_kinds = ("binary", "aggregate", "count_if", "arg_extremum")
    for depth in range(GRAMMAR_MAX_DEPTH + 1):
        variants = [_grammar_ref("cell"), _grammar_ref("literal")]
        if depth:
            for kind in recursive_kinds:
                definition_name = f"{kind}_{depth}"
                grammar_defs[definition_name] = _grammar_variant(
                    canonical_defs[kind], depth=depth - 1
                )
                variants.append(_grammar_ref(definition_name))
        grammar_defs[f"expression_{depth}"] = {"oneOf": variants}

    properties["program"] = _grammar_ref(f"expression_{GRAMMAR_MAX_DEPTH}")
    schema["$defs"] = grammar_defs
    return schema


# Keep the canonical recursive schema for validation and provenance. vLLM gets
# an equivalent acyclic schema because recursive XGrammar compilation can stall.
PROGRAM_GRAMMAR_SCHEMA = _build_program_grammar_schema()


def _object(
    raw: object,
    *,
    required: set[str],
    depth: int,
    optional: frozenset[str] = frozenset(),
) -> dict[str, object]:
    if depth > PARSER_MAX_DEPTH:
        raise ValueError("Program exceeds maximum nesting depth")
    if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
        raise TypeError("Every program node must be an object")
    node = cast(dict[str, object], raw)
    keys = set(node)
    if not required <= keys or keys - required - optional:
        raise ValueError(
            f"Program node fields mismatch; expected={sorted(required)}, actual={sorted(keys)}"
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
    if not isinstance(value, list) or not value or len(value) > PARSER_MAX_ITEMS:
        raise ValueError(f"{field} must contain 1..{PARSER_MAX_ITEMS} nodes")
    return value


def expression_from_dict(raw: object, *, _depth: int = 0) -> ScalarExpr:
    if not isinstance(raw, dict):
        raise TypeError("Program node must be an object")
    kind = raw.get("kind")
    if kind == "cell":
        node = _object(
            raw,
            required={"kind", "variable", "row_index", "column_index"},
            optional=CELL_LINEAGE_FIELDS,
            depth=_depth,
        )
        variable = node["variable"]
        if not isinstance(variable, str) or not _VARIABLE_RE.fullmatch(variable):
            raise ValueError("variable must match dfN")
        # Both fields are settled from the evidence's source unit whenever it has one, so a
        # cell may leave them out. Emitting them costs about four tokens of every ten a
        # cohort program spends, and a cohort program is what runs closest to the clock.
        value_column = _enum(
            node.get("value_column", "base_value"),
            {"numeric_value", "base_value"},
            field="value_column",
        )
        dimension = cast(
            Dimension,
            _enum(node.get("dimension", "UNKNOWN"), set(DIMENSIONS), field="dimension"),
        )
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
