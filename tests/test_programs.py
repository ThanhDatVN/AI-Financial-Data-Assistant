from __future__ import annotations

import pandas as pd
import pytest

from vifinqa.programs.compiler import compile_expression
from vifinqa.programs.dimensions import infer_dimension
from vifinqa.programs.executor import execute_expression, execute_expression_isolated
from vifinqa.programs.grounding import prepare_program, validate_query_coverage
from vifinqa.programs.ir import (
    AggregateExpr,
    ArgExtremumExpr,
    BinaryExpr,
    CellExpr,
    CountIfExpr,
    LiteralExpr,
)


def test_compiled_ir_executes_against_long_evidence() -> None:
    frame = pd.DataFrame(
        {
            "row_index": [0, 0, 1, 1],
            "column_index": [0, 1, 0, 1],
            "base_value": [None, None, None, 10.0],
        }
    )
    ir = BinaryExpr("/", CellExpr("df1", 1, 1), LiteralExpr(2.0))
    query = compile_expression(ir)
    assert execute_expression(query, {"df1": frame}) == 5.0


def test_executor_rejects_imports_and_dunder_access() -> None:
    with pytest.raises((SyntaxError, ValueError)):
        execute_expression("__import__('os').system('whoami')", {})
    with pytest.raises(ValueError, match="Private/dunder"):
        execute_expression("df.__class__", {"df": pd.DataFrame()})


def test_isolated_executor_returns_scalar_and_enforces_timeout() -> None:
    assert execute_expression_isolated("1 + 2", {}, timeout_seconds=5.0) == 3.0
    with pytest.raises(TimeoutError):
        execute_expression_isolated("1 + 2", {}, timeout_seconds=1e-9)


def test_aggregate_count_and_argmax_compile_to_executable_expressions() -> None:
    aggregate = AggregateExpr("mean", (LiteralExpr(2), LiteralExpr(4), LiteralExpr(6)))
    count = CountIfExpr((LiteralExpr(-1), LiteralExpr(2), LiteralExpr(3)), ">", LiteralExpr(0))
    selected_value = ArgExtremumExpr(
        "argmax",
        keys=(LiteralExpr(3), LiteralExpr(8), LiteralExpr(5)),
        values=(LiteralExpr(30), LiteralExpr(80), LiteralExpr(50)),
    )
    assert execute_expression(compile_expression(aggregate), {}) == 4.0
    assert execute_expression(compile_expression(count), {}) == 2.0
    assert execute_expression(compile_expression(selected_value), {}) == 80.0


def test_argmax_tie_policy_selects_first_value() -> None:
    expression = ArgExtremumExpr(
        "argmax",
        keys=(LiteralExpr(8), LiteralExpr(8)),
        values=(LiteralExpr(1), LiteralExpr(2)),
    )
    assert execute_expression(compile_expression(expression), {}) == 1.0


def test_dimension_checker_accepts_ratio_and_rejects_incompatible_sum() -> None:
    ratio = BinaryExpr(
        "/",
        CellExpr("df1", 0, 0, dimension="VND"),
        CellExpr("df2", 0, 0, dimension="VND"),
    )
    assert infer_dimension(ratio) == "RATIO"
    incompatible = BinaryExpr(
        "+",
        CellExpr("df1", 0, 0, dimension="VND"),
        CellExpr("df2", 0, 0, dimension="PERCENT"),
    )
    with pytest.raises(ValueError, match="Incompatible dimensions"):
        infer_dimension(incompatible)


def test_dimension_checker_tracks_count_and_argmax_value_dimension() -> None:
    count = CountIfExpr(
        (CellExpr("df1", 0, 0, dimension="PERCENT"),), "<", LiteralExpr(0, "PERCENT")
    )
    argmax = ArgExtremumExpr(
        "argmax",
        keys=(CellExpr("df1", 0, 0, dimension="VND"),),
        values=(CellExpr("df1", 1, 0, dimension="PERCENT"),),
    )
    assert infer_dimension(count) == "COUNT"
    assert infer_dimension(argmax) == "PERCENT"


def test_grounding_checks_cell_unit_and_applies_target_divisor() -> None:
    frame = pd.DataFrame(
        {
            "row_index": [0],
            "column_index": [1],
            "source_unit": ["VND"],
            "base_value": [2_000_000.0],
            "numeric_value": [2_000_000.0],
        }
    )
    expression = CellExpr("df1", 0, 1, dimension="VND")
    prepared, dimension = prepare_program(
        expression,
        selected_variables=["df1"],
        frames={"df1": frame},
        target_unit="MILLION_VND",
        target_divisor=1_000_000.0,
    )
    assert dimension == "VND"
    assert execute_expression(compile_expression(prepared), {"df1": frame}) == 2.0


def test_panel_coverage_requires_entities_and_accepts_labeled_prior_year() -> None:
    def evidence(ticker: str, report_year: int, column_label: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "row_index": [0],
                "column_index": [1],
                "ticker": [ticker],
                "report_year": [report_year],
                "column_label": [column_label],
                "source_unit": ["VND"],
                "base_value": [1.0],
                "numeric_value": [1.0],
            }
        )

    frames = {
        "df1": evidence("AAA", 2022, "Số đầu năm"),
        "df2": evidence("BBB", 2022, "2022"),
    }
    expression = AggregateExpr(
        "sum",
        (
            CellExpr("df1", 0, 1, dimension="VND"),
            CellExpr("df2", 0, 1, dimension="VND"),
        ),
    )
    validate_query_coverage(
        expression,
        frames=frames,
        required_tickers=["AAA", "BBB"],
        required_years=[2021, 2022],
    )
    with pytest.raises(ValueError, match="required tickers"):
        validate_query_coverage(
            CellExpr("df1", 0, 1, dimension="VND"),
            frames={"df1": frames["df1"]},
            required_tickers=["AAA", "BBB"],
            required_years=[2021],
        )
