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
    Comparator,
    Condition,
    CountIfExpr,
    Dimension,
    LiteralExpr,
    ScalarExpr,
    SelectExpr,
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


class RankMismatchError(ValueError):
    """A ranked selection whose key list does not line up with its members.

    Separate from the other refusals because it has a reading that runs: rank the members
    themselves, which is already what a missing key list means. Worth telling the model about
    while it can still answer, and not worth losing the question over once it cannot.
    """


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
                {"$ref": "#/$defs/select"},
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
        "condition": {
            "type": "object",
            "properties": {
                "left": {
                    "type": "array",
                    "items": _EXPRESSION_REF,
                    "minItems": 1,
                    "maxItems": 100,
                },
                "comparator": {"enum": ["<", "<=", ">", ">=", "==", "!="]},
                "right": _EXPRESSION_REF,
                "right_per_member": {
                    "type": "array",
                    "items": _EXPRESSION_REF,
                    "minItems": 1,
                    "maxItems": 100,
                },
            },
            "required": ["left", "comparator"],
            "additionalProperties": False,
        },
        "select": {
            "type": "object",
            "properties": {
                "kind": {"const": "select"},
                "operator": {
                    "enum": ["sum", "mean", "min", "max", "median", "count", "argmin", "argmax"]
                },
                "members": {
                    "type": "array",
                    "items": _EXPRESSION_REF,
                    "minItems": 1,
                    "maxItems": 100,
                },
                "conditions": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/condition"},
                    "maxItems": 8,
                },
                "keys": {
                    "type": "array",
                    "items": _EXPRESSION_REF,
                    "minItems": 1,
                    "maxItems": 100,
                },
            },
            "required": ["kind", "operator", "members"],
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
GRAMMAR_MAX_DEPTH = 5
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
    recursive_kinds = ("binary", "aggregate", "count_if", "arg_extremum", "select")
    for depth in range(GRAMMAR_MAX_DEPTH + 1):
        variants = [_grammar_ref("cell"), _grammar_ref("literal")]
        if depth:
            condition_name = f"condition_{depth}"
            grammar_defs[condition_name] = _grammar_variant(
                canonical_defs["condition"], depth=depth - 1
            )
            for kind in recursive_kinds:
                definition_name = f"{kind}_{depth}"
                definition = cast(
                    dict[str, object], _grammar_variant(canonical_defs[kind], depth=depth - 1)
                )
                if kind == "select":
                    select_properties = cast(dict[str, object], definition["properties"])
                    conditions = cast(dict[str, object], select_properties["conditions"])
                    conditions["items"] = _grammar_ref(condition_name)
                grammar_defs[definition_name] = definition
                variants.append(_grammar_ref(definition_name))
        grammar_defs[f"expression_{depth}"] = {"oneOf": variants}

    properties["program"] = _grammar_ref(f"expression_{GRAMMAR_MAX_DEPTH}")
    schema["$defs"] = grammar_defs
    return schema


# Keep the canonical recursive schema for validation and provenance. vLLM gets
# an equivalent acyclic schema because recursive XGrammar compilation can stall.
PROGRAM_GRAMMAR_SCHEMA = _build_program_grammar_schema()


# Which root nodes can possibly answer a question, given the unit the answer is in.
#
# Measured 19/08/2026 on 499 dev samples with the gold tables already in the prompt and no
# distractors -- retrieval difficulty removed entirely. The boundary was not difficulty, it was
# the dimension of the answer:
#
#     lookup       target currency   root cell           pass@1 0.8371
#     conditional  target currency   root select         pass@1 0.1274
#     ratio        target PERCENT    root binary         pass@1 0.0345
#     change       target PERCENT    root binary         pass@1 0.0000
#     extremum     target YEAR       root arg_extremum   pass@1 0.0000
#
# Two exact zeros, from a model that answers 84% of plain lookups. Replaying the gold programs
# through this module and the compiler reproduced all 499 recorded answers exactly, so neither the
# unit convention nor the tolerance is at fault. Across three scored runs the model emitted
# `arg_extremum` 0 times in 519, 640 and 631 clean programs.
#
# Only YEAR is narrowed by default, and the restraint is the point:
#
#  * YEAR earns it. 65 failed attempts in run 3226 died on "Program dimension VND is incompatible
#    with target YEAR", which says the program returned an amount. A year is a label, never a
#    stored value, so no legitimate answer is a bare cell -- checked against all 99 gold extremum
#    samples, every one an `arg_extremum`.
#  * PERCENT does not, on this evidence. Six gold `lookup` samples have target PERCENT and a plain
#    `cell` root, because their source column already holds a percentage -- forbidding cell there
#    would make correct answers unreachable. And `select` is permissive enough to return a VND
#    amount anyway, so narrowing to (binary, select, cell) removes almost nothing. `percent` is
#    kept selectable so it can be measured as its own variable rather than assumed.
#
# Narrowing the ROOT alone is also deliberate. Sub-expressions keep the whole grammar, so a binary
# root still holds cells and a select root still holds whatever it needs; only the single decision
# the measurement indicts is taken away.
ROOT_GRAMMAR_POLICIES: dict[str, dict[str, tuple[str, ...]]] = {
    # Today's behaviour: the model picks any root for any target.
    "off": {},
    # The answer is a label, so it has to come from picking among candidates.
    "year": {"YEAR": ("arg_extremum", "select")},
    # Pushes the one node three scored runs never produced. Competing hypothesis to "year": if the
    # model cannot write `arg_extremum` at all, this reads worse than "year" rather than better.
    "year-strict": {"YEAR": ("arg_extremum",)},
    # PERCENT loses `select`, on evidence rather than on the guess the first draft made.
    # Run dev_year measured what the model writes for a percentage change: `select` 116 times out
    # of 116, where the gold program is a binary division times one hundred. And of the 122 gold
    # PERCENT/RATIO samples, 116 have a binary root and 6 a cell root -- not one is a select. So
    # forbidding it here costs no reachable answer and forbids exactly the mistake.
    "year+percent": {
        "YEAR": ("arg_extremum", "select"),
        "PERCENT": ("binary", "cell"),
        "RATIO": ("binary", "cell"),
    },
    # Everything the evidence supports at once. `select` is gone from both targets: for YEAR
    # because the model took it 198 times out of 198 even when `arg_extremum` was the only other
    # option, and filled its members with cells every time (765 of 765), which returns an amount
    # where a year was asked for -- 190 of those 198 attempts died saying exactly that.
    "strict": {
        "YEAR": ("arg_extremum",),
        "PERCENT": ("binary", "cell"),
        "RATIO": ("binary", "cell"),
    },
}
DEFAULT_ROOT_GRAMMAR_POLICY = "off"


def program_grammar_for_target(
    target_unit: str, *, policy: str = DEFAULT_ROOT_GRAMMAR_POLICY
) -> dict[str, object]:
    """The decoding grammar with the root narrowed to the shapes this target admits.

    Returns the unrestricted grammar for `policy="off"` and for any target the policy does not
    name, so currency, share and count questions decode exactly as they do today -- the 84% case
    is never touched by this.
    """
    if policy not in ROOT_GRAMMAR_POLICIES:
        raise ValueError(
            f"Unknown root grammar policy {policy!r}; "
            f"expected one of {sorted(ROOT_GRAMMAR_POLICIES)}"
        )
    kinds = ROOT_GRAMMAR_POLICIES[policy].get(target_unit.upper())
    if not kinds:
        return PROGRAM_GRAMMAR_SCHEMA
    schema = copy.deepcopy(PROGRAM_GRAMMAR_SCHEMA)
    properties = cast(dict[str, object], schema["properties"])
    defined = cast(dict[str, object], schema["$defs"])
    names = [
        f"{kind}_{GRAMMAR_MAX_DEPTH}" if kind not in {"cell", "literal"} else kind for kind in kinds
    ]
    missing = [name for name in names if name not in defined]
    if missing:  # pragma: no cover - guards a future rename of the unrolled definitions
        raise ValueError(f"Root grammar references undefined nodes: {missing}")
    variants = [_grammar_ref(name) for name in names]
    properties["program"] = variants[0] if len(variants) == 1 else {"oneOf": variants}
    return schema


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


def _conditions(value: object) -> list[object]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > 8:
        raise ValueError("conditions must contain at most 8 entries")
    return value


def _condition_from_dict(
    raw: object, *, member_count: int, depth: int, lenient: bool = False
) -> Condition:
    node = _object(
        raw,
        required={"left", "comparator"},
        optional=frozenset({"right", "right_per_member"}),
        depth=depth,
    )
    left = tuple(
        expression_from_dict(item, lenient=lenient, _depth=depth)
        for item in _items(node["left"], field="left")
    )
    if len(left) != member_count:
        # The message goes straight back to the model as retry feedback, so it has to say what
        # to change. Question 473 spent all three attempts on the wordless version of this and
        # returned the same mismatch every time.
        raise ValueError(
            f"Condition operands must align with the cohort members: the select node has "
            f"{member_count} members but this condition lists {len(left)} entries in 'left'. "
            f"Give 'left' exactly {member_count} entries, one per member, in the same order."
        )
    comparator = _enum(node["comparator"], {"<", "<=", ">", ">=", "==", "!="}, field="comparator")
    shared, per_member = node.get("right"), node.get("right_per_member")
    if (shared is None) == (per_member is None):
        raise ValueError("A condition needs exactly one of right or right_per_member")
    right: ScalarExpr | tuple[ScalarExpr, ...]
    if per_member is not None:
        thresholds = tuple(
            expression_from_dict(item, lenient=lenient, _depth=depth)
            for item in _items(per_member, field="right_per_member")
        )
        if len(thresholds) != member_count:
            raise ValueError("Per-member thresholds must align with the cohort members")
        right = thresholds
    else:
        right = expression_from_dict(shared, lenient=lenient, _depth=depth)
    return Condition(left, cast(Comparator, comparator), right)


def expression_from_dict(raw: object, *, lenient: bool = False, _depth: int = 0) -> ScalarExpr:
    """Read a program off the wire.

    `lenient` is for the last attempt only. It reads a key list whose length disagrees with the
    members as no key list at all -- the reading a missing one already gets, and the one thing
    the model plainly did not mean is for the question to go unanswered. 31 questions in the
    full run ended on this and were then handed to a keyword-matched single cell, which ranks
    nothing at all.
    """
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
            expression_from_dict(node["left"], lenient=lenient, _depth=_depth + 1),
            expression_from_dict(node["right"], lenient=lenient, _depth=_depth + 1),
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
                expression_from_dict(item, lenient=lenient, _depth=_depth + 1)
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
                expression_from_dict(item, lenient=lenient, _depth=_depth + 1)
                for item in _items(node["operands"], field="operands")
            ),
            cast(Literal["<", "<=", ">", ">=", "==", "!="], comparator),
            expression_from_dict(node["threshold"], lenient=lenient, _depth=_depth + 1),
        )
    if kind == "select":
        node = _object(
            raw,
            required={"kind", "operator", "members"},
            optional=frozenset({"conditions", "keys"}),
            depth=_depth,
        )
        operator = _enum(
            node["operator"],
            {"sum", "mean", "min", "max", "median", "count", "argmin", "argmax"},
            field="operator",
        )
        members = tuple(
            expression_from_dict(item, lenient=lenient, _depth=_depth + 1)
            for item in _items(node["members"], field="members")
        )
        raw_keys = node.get("keys")
        keys = (
            None
            if raw_keys is None
            else tuple(
                expression_from_dict(item, lenient=lenient, _depth=_depth + 1)
                for item in _items(raw_keys, field="keys")
            )
        )
        # The compiler already reads a missing key list as "rank the members themselves" and
        # ignores keys an operator cannot rank with. Rejecting those here undid that: a program
        # never reached the compiler, so relaxing the rule there alone changed nothing. A key
        # list whose length disagrees with the members is still a real contradiction.
        if operator in {"argmin", "argmax"}:
            if keys is not None and len(keys) != len(members):
                if not lenient:
                    raise RankMismatchError(
                        f"Ranked selection needs one key per member: {len(members)} members "
                        f"but {len(keys)} keys."
                    )
                keys = None
        elif keys is not None:
            keys = None
        conditions = tuple(
            _condition_from_dict(item, member_count=len(members), depth=_depth + 1, lenient=lenient)
            for item in _conditions(node.get("conditions"))
        )
        if operator == "median" and conditions:
            raise ValueError("Median over a filtered cohort is not supported")
        return SelectExpr(
            cast(
                Literal["sum", "mean", "min", "max", "median", "count", "argmin", "argmax"],
                operator,
            ),
            members,
            conditions,
            keys,
        )
    if kind == "arg_extremum":
        node = _object(
            raw,
            required={"kind", "mode", "keys", "values"},
            depth=_depth,
        )
        mode = _enum(node["mode"], {"argmin", "argmax"}, field="mode")
        keys = tuple(
            expression_from_dict(item, lenient=lenient, _depth=_depth + 1)
            for item in _items(node["keys"], field="keys")
        )
        values = tuple(
            expression_from_dict(item, lenient=lenient, _depth=_depth + 1)
            for item in _items(node["values"], field="values")
        )
        if len(keys) != len(values):
            raise ValueError("arg_extremum keys and values must have equal length")
        return ArgExtremumExpr(cast(Literal["argmin", "argmax"], mode), keys, values)
    raise ValueError(f"Unknown program node kind: {kind!r}")


def expression_to_dict(expression: ScalarExpr) -> dict[str, object]:
    """Write a program back out in the wire shape `expression_from_dict` reads.

    `dataclasses.asdict` looks like it does this and does not: it recurses into children as plain
    dicts and drops the type of every one of them, so only a single-node program survives the
    trip. The synthetic sampler used it, and four of its five families wrote a `program` field
    that could never be read back -- unnoticed, because nothing had tried to read one yet.

    Kept beside the reader so the two stay in step; `test_program_round_trips_through_serde`
    fails if either side learns a node the other does not.
    """
    if isinstance(expression, CellExpr):
        return {
            "kind": "cell",
            "variable": expression.variable,
            "row_index": expression.row_index,
            "column_index": expression.column_index,
            "value_column": expression.value_column,
            "dimension": expression.dimension,
        }
    if isinstance(expression, LiteralExpr):
        return {"kind": "literal", "value": expression.value, "dimension": expression.dimension}
    if isinstance(expression, BinaryExpr):
        return {
            "kind": "binary",
            "operator": expression.operator,
            "left": expression_to_dict(expression.left),
            "right": expression_to_dict(expression.right),
        }
    if isinstance(expression, AggregateExpr):
        return {
            "kind": "aggregate",
            "operator": expression.operator,
            "operands": [expression_to_dict(item) for item in expression.operands],
        }
    if isinstance(expression, CountIfExpr):
        return {
            "kind": "count_if",
            "operands": [expression_to_dict(item) for item in expression.operands],
            "comparator": expression.comparator,
            "threshold": expression_to_dict(expression.threshold),
        }
    if isinstance(expression, SelectExpr):
        node: dict[str, object] = {
            "kind": "select",
            "operator": expression.operator,
            "members": [expression_to_dict(item) for item in expression.members],
        }
        if expression.conditions:
            node["conditions"] = [
                _condition_to_dict(condition) for condition in expression.conditions
            ]
        if expression.keys is not None:
            node["keys"] = [expression_to_dict(item) for item in expression.keys]
        return node
    if isinstance(expression, ArgExtremumExpr):
        return {
            "kind": "arg_extremum",
            "mode": expression.mode,
            "keys": [expression_to_dict(item) for item in expression.keys],
            "values": [expression_to_dict(item) for item in expression.values],
        }
    raise TypeError(f"Unsupported expression: {type(expression).__name__}")


def _condition_to_dict(condition: Condition) -> dict[str, object]:
    node: dict[str, object] = {
        "left": [expression_to_dict(item) for item in condition.left],
        "comparator": condition.comparator,
    }
    if isinstance(condition.right, tuple):
        node["right_per_member"] = [expression_to_dict(item) for item in condition.right]
    else:
        node["right"] = expression_to_dict(condition.right)
    return node
