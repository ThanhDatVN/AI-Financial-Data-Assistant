from __future__ import annotations

import jsonschema
import pytest

from vifinqa.programs.compiler import compile_expression
from vifinqa.programs.ir import BinaryExpr, CellExpr
from vifinqa.programs.serde import (
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

    restored = dict(grammar_variables)
    restored["uniqueItems"] = True
    assert restored == canonical_variables


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
