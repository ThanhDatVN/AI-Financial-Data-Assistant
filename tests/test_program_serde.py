from __future__ import annotations

import json

import jsonschema
import pytest

from vifinqa.programs.compiler import compile_expression
from vifinqa.programs.ir import BinaryExpr, CellExpr
from vifinqa.programs.serde import (
    GRAMMAR_MAX_DEPTH,
    GRAMMAR_MAX_ITEMS,
    MAX_ROUTE_FAN_OUT,
    PARSER_MAX_DEPTH,
    PARSER_MAX_ITEMS,
    PROGRAM_GRAMMAR_SCHEMA,
    PROGRAM_JSON_SCHEMA,
    expression_from_dict,
)


def _sample_program() -> dict[str, object]:
    return {
        "selected_variables": ["df1"],
        "program": {
            "kind": "binary",
            "operator": "/",
            "left": {
                "kind": "cell",
                "variable": "df1",
                "row_index": 2,
                "column_index": 1,
                "value_column": "base_value",
                "dimension": "VND",
            },
            "right": {
                "kind": "literal",
                "value": 2,
                "dimension": "DIMENSIONLESS",
            },
        },
    }


def test_recursive_program_schema_and_parser_accept_typed_tree() -> None:
    payload = _sample_program()
    jsonschema.validate(payload, PROGRAM_JSON_SCHEMA)
    expression = expression_from_dict(payload["program"])
    assert isinstance(expression, BinaryExpr)
    assert isinstance(expression.left, CellExpr)
    assert "df1.loc" in compile_expression(expression)


def test_vllm_grammar_schema_omits_only_backend_unsupported_uniqueness() -> None:
    properties = PROGRAM_JSON_SCHEMA["properties"]
    grammar_properties = PROGRAM_GRAMMAR_SCHEMA["properties"]
    assert isinstance(properties, dict)
    assert isinstance(grammar_properties, dict)
    canonical_variables = properties["selected_variables"]
    grammar_variables = grammar_properties["selected_variables"]
    assert isinstance(canonical_variables, dict)
    assert isinstance(grammar_variables, dict)
    assert canonical_variables["uniqueItems"] is True
    assert "uniqueItems" not in grammar_variables

    for key, value in canonical_variables.items():
        if key != "uniqueItems":
            assert grammar_variables[key] == value


def _schema_refs(value: object) -> list[str]:
    if isinstance(value, dict):
        refs = [str(value["$ref"])] if "$ref" in value else []
        return refs + [ref for item in value.values() for ref in _schema_refs(item)]
    if isinstance(value, list):
        return [ref for item in value for ref in _schema_refs(item)]
    return []


def test_vllm_grammar_schema_is_acyclic() -> None:
    grammar_defs = PROGRAM_GRAMMAR_SCHEMA["$defs"]
    assert isinstance(grammar_defs, dict)
    assert f"expression_{GRAMMAR_MAX_DEPTH}" in grammar_defs
    assert "expression" not in grammar_defs

    graph: dict[str, set[str]] = {}
    for name, definition in grammar_defs.items():
        graph[name] = {ref.removeprefix("#/$defs/") for ref in _schema_refs(definition)}
        assert graph[name].issubset(grammar_defs)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        assert name not in visiting, f"Recursive grammar reference detected at {name}"
        if name in visited:
            return
        visiting.add(name)
        for child in graph[name]:
            visit(child)
        visiting.remove(name)
        visited.add(name)

    for definition_name in graph:
        visit(definition_name)

    jsonschema.validate(_sample_program(), PROGRAM_GRAMMAR_SCHEMA)


_LITERAL: dict[str, object] = {"kind": "literal", "value": 1, "dimension": "DIMENSIONLESS"}


def _nested_binary(depth: int) -> dict[str, object]:
    program = _LITERAL
    for _ in range(depth):
        program = {"kind": "binary", "operator": "+", "left": program, "right": _LITERAL}
    return program


def test_vllm_grammar_budget_is_reachable_and_stays_inside_the_parser_budget() -> None:
    assert GRAMMAR_MAX_DEPTH < PARSER_MAX_DEPTH
    assert GRAMMAR_MAX_ITEMS < PARSER_MAX_ITEMS

    at_budget = _nested_binary(GRAMMAR_MAX_DEPTH)
    jsonschema.validate(
        {"selected_variables": ["df1"], "program": at_budget}, PROGRAM_GRAMMAR_SCHEMA
    )
    expression_from_dict(at_budget)

    beyond_grammar = _nested_binary(GRAMMAR_MAX_DEPTH + 1)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            {"selected_variables": ["df1"], "program": beyond_grammar}, PROGRAM_GRAMMAR_SCHEMA
        )
    # The parser keeps its own, wider budget: the decoder is constrained, not the validator.
    expression_from_dict(beyond_grammar)


def test_parser_still_enforces_its_own_depth_and_width_budget() -> None:
    expression_from_dict(_nested_binary(PARSER_MAX_DEPTH))
    with pytest.raises(ValueError, match="maximum nesting depth"):
        expression_from_dict(_nested_binary(PARSER_MAX_DEPTH + 1))

    expression_from_dict(
        {"kind": "aggregate", "operator": "sum", "operands": [_LITERAL] * PARSER_MAX_ITEMS}
    )
    with pytest.raises(ValueError, match="nodes"):
        expression_from_dict(
            {
                "kind": "aggregate",
                "operator": "sum",
                "operands": [_LITERAL] * (PARSER_MAX_ITEMS + 1),
            }
        )


def test_vllm_grammar_admits_the_widest_question_in_the_release() -> None:
    # Routing fans out to tickers x years, so a cohort program over the widest question
    # needs one operand per route. Losing those means losing the hardest questions.
    assert GRAMMAR_MAX_ITEMS >= MAX_ROUTE_FAN_OUT
    cohort = [_LITERAL] * MAX_ROUTE_FAN_OUT
    payload = {
        "selected_variables": ["df1"],
        "program": {
            "kind": "arg_extremum",
            "mode": "argmax",
            "keys": cohort,
            "values": cohort,
        },
    }
    jsonschema.validate(payload, PROGRAM_GRAMMAR_SCHEMA)
    expression_from_dict(payload["program"])


def test_vllm_grammar_stays_small_enough_to_compile() -> None:
    # A grammar unrolled to the parser budget reached 103 definitions holding 80 arrays of
    # 100 items, and XGrammar compilation stalled before emitting a single token.
    grammar_defs = PROGRAM_GRAMMAR_SCHEMA["$defs"]
    assert isinstance(grammar_defs, dict)
    assert len(grammar_defs) <= 40
    serialized = json.dumps(PROGRAM_GRAMMAR_SCHEMA)
    assert len(serialized) <= 16_000
    assert f'"maxItems": {PARSER_MAX_ITEMS}' not in serialized
    assert serialized.count(f'"maxItems": {GRAMMAR_MAX_ITEMS}') == 4 * GRAMMAR_MAX_DEPTH
    # Bounded repetitions are what the grammar compiler expands, so cap the total.
    assert 4 * GRAMMAR_MAX_DEPTH * GRAMMAR_MAX_ITEMS <= 1_000


def test_program_parser_rejects_extra_fields_and_unequal_arg_extremum() -> None:
    cell = _sample_program()["program"]
    assert isinstance(cell, dict)
    cell["unexpected"] = True
    with pytest.raises(ValueError, match="fields mismatch"):
        expression_from_dict(cell)

    with pytest.raises(ValueError, match="equal length"):
        expression_from_dict(
            {
                "kind": "arg_extremum",
                "mode": "argmax",
                "keys": [{"kind": "literal", "value": 1, "dimension": "DIMENSIONLESS"}],
                "values": [
                    {"kind": "literal", "value": 2, "dimension": "DIMENSIONLESS"},
                    {"kind": "literal", "value": 3, "dimension": "DIMENSIONLESS"},
                ],
            }
        )


def test_program_parser_rejects_non_finite_literal_and_bad_variable() -> None:
    with pytest.raises(ValueError, match="finite"):
        expression_from_dict(
            {"kind": "literal", "value": float("nan"), "dimension": "DIMENSIONLESS"}
        )
    with pytest.raises(ValueError, match="dfN"):
        expression_from_dict(
            {
                "kind": "cell",
                "variable": "../../escape",
                "row_index": 0,
                "column_index": 0,
                "value_column": "base_value",
                "dimension": "VND",
            }
        )
