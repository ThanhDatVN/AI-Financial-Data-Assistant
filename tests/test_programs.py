from __future__ import annotations

import pandas as pd
import pytest

from vifinqa.programs.compiler import compile_expression
from vifinqa.programs.dimensions import infer_dimension
from vifinqa.programs.executor import execute_expression, execute_expression_isolated
from vifinqa.programs.grounding import (
    normalize_cells,
    prepare_program,
    validate_answer_plausibility,
    validate_query_coverage,
)
from vifinqa.programs.ir import (
    AggregateExpr,
    ArgExtremumExpr,
    BinaryExpr,
    CellExpr,
    Condition,
    CountIfExpr,
    LiteralExpr,
    SelectExpr,
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


def test_grounding_points_cells_at_the_column_their_unit_dictates() -> None:
    frames = {
        "df1": pd.DataFrame(
            {
                "row_index": [0],
                "column_index": [1],
                "source_unit": ["MILLION_VND"],
                "numeric_value": [7.0],
                "base_value": [7_000_000.0],
            }
        ),
        "df2": pd.DataFrame(
            {
                "row_index": [0],
                "column_index": [1],
                "source_unit": ["PERCENT"],
                "numeric_value": [12.5],
                "base_value": [12.5],
            }
        ),
    }
    # A model that guessed numeric_value on a scaled amount used to lose the whole question.
    scaled = CellExpr("df1", 0, 1, value_column="numeric_value", dimension="VND")
    prepared, _ = prepare_program(
        scaled,
        selected_variables=["df1"],
        frames={"df1": frames["df1"]},
        target_unit="MILLION_VND",
        target_divisor=1_000_000.0,
    )
    assert execute_expression(compile_expression(prepared), frames) == 7.0

    ratio = CellExpr("df2", 0, 1, value_column="base_value", dimension="PERCENT")
    normalized = normalize_cells(ratio, frames)
    assert isinstance(normalized, CellExpr)
    assert normalized.value_column == "numeric_value"

    # A model that labelled a million-VND cell a percentage used to repeat that claim until
    # its attempts ran out. The source unit already settles the question.
    mislabelled = CellExpr("df1", 0, 1, value_column="numeric_value", dimension="PERCENT")
    corrected = normalize_cells(mislabelled, frames)
    assert isinstance(corrected, CellExpr)
    assert (corrected.dimension, corrected.value_column) == ("VND", "base_value")

    # Without lineage the claim stands, and validation still refuses a bare currency claim.
    unknown = pd.DataFrame(
        {
            "row_index": [0],
            "column_index": [1],
            "source_unit": ["UNKNOWN"],
            "numeric_value": [3.0],
            "base_value": [3.0],
        }
    )
    claimed = CellExpr("df3", 0, 1, dimension="COUNT")
    assert normalize_cells(claimed, {"df3": unknown}) == claimed


def test_grounding_refuses_a_multi_cell_program_that_cancels_to_zero() -> None:
    # A statement restates the same figure with the opposite sign, and adding the two passes
    # every unit, coverage and execution check while answering zero.
    cancelling = BinaryExpr(
        "+",
        CellExpr("df1", 0, 1, dimension="VND"),
        CellExpr("df2", 5, 2, dimension="VND"),
    )
    with pytest.raises(ValueError, match="exactly zero"):
        validate_answer_plausibility(0.0, cancelling)

    validate_answer_plausibility(0.0, CellExpr("df1", 0, 1, dimension="VND"))
    validate_answer_plausibility(1.0, cancelling)


def _cohort(values: list[list[float]]) -> dict[str, pd.DataFrame]:
    return {
        f"df{index}": pd.DataFrame(
            {
                "row_index": list(range(len(column))),
                "column_index": [1] * len(column),
                "source_unit": ["UNKNOWN"] * len(column),
                "numeric_value": column,
                "base_value": column,
            }
        )
        for index, column in enumerate(values, start=1)
    }


def test_selection_sums_only_the_members_that_pass_every_condition() -> None:
    # "Tổng doanh thu 2022 của các công ty có biên lợi nhuận dương trong cả ba năm."
    # Counting a predicate was never the hard part; selecting by one was.
    frames = _cohort([[10.0, 1.0, 1.0, 1.0], [20.0, 1.0, -1.0, 1.0], [30.0, 2.0, 2.0, 2.0]])
    members = tuple(CellExpr(f"df{index}", 0, 1) for index in (1, 2, 3))
    conditions = tuple(
        Condition(
            left=tuple(CellExpr(f"df{index}", year, 1) for index in (1, 2, 3)),
            comparator=">",
            right=LiteralExpr(0.0),
        )
        for year in (1, 2, 3)
    )
    program = SelectExpr("sum", members, conditions)
    assert execute_expression(compile_expression(program), frames) == 40.0

    assert execute_expression(compile_expression(SelectExpr("count", members, conditions)), frames)
    assert infer_dimension(SelectExpr("count", members, conditions)) == "COUNT"


def test_selection_ranks_within_the_half_below_the_median() -> None:
    # "Trong nhóm có D/E dưới trung vị, doanh nghiệp tăng trưởng cao nhất." The median is the
    # same node with nothing to filter by, so no separate operator is needed.
    frames = _cohort(
        [[100.0, 1.0, 5.0], [200.0, 2.0, 9.0], [300.0, 3.0, 1.0], [400.0, 4.0, 2.0]]
        + [[500.0, 5.0, 3.0]]
    )
    leverage = tuple(CellExpr(f"df{index}", 1, 1) for index in range(1, 6))
    median = SelectExpr("median", leverage)
    assert execute_expression(compile_expression(median), frames) == 3.0

    program = SelectExpr(
        "argmax",
        members=tuple(CellExpr(f"df{index}", 0, 1) for index in range(1, 6)),
        conditions=(Condition(left=leverage, comparator="<", right=median),),
        keys=tuple(CellExpr(f"df{index}", 2, 1) for index in range(1, 6)),
    )
    assert execute_expression(compile_expression(program), frames) == 200.0

    # Naming shared work once is what keeps this from expanding past a megabyte.
    compiled = compile_expression(program)
    assert compiled.startswith("((_v0 :=") and compiled.endswith(")[-1]")
    assert len(compiled) < 20_000


def test_selection_refuses_an_empty_subset_rather_than_inventing_an_extreme() -> None:
    frames = _cohort([[10.0, 1.0], [20.0, 2.0]])
    impossible = Condition(
        left=(CellExpr("df1", 1, 1), CellExpr("df2", 1, 1)),
        comparator="<",
        right=LiteralExpr(0.0),
    )
    program = SelectExpr(
        "max", members=(CellExpr("df1", 0, 1), CellExpr("df2", 0, 1)), conditions=(impossible,)
    )
    with pytest.raises(ValueError, match="not finite"):
        execute_expression(compile_expression(program), frames)


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
