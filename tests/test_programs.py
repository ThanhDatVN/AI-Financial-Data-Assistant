from __future__ import annotations

import json

import pandas as pd
import pytest

from vifinqa.programs.compiler import MAX_QUERY_CHARS, compile_expression
from vifinqa.programs.dimensions import infer_dimension
from vifinqa.programs.executor import execute_expression, execute_expression_isolated
from vifinqa.programs.grounding import (
    SoftGroundingError,
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
from vifinqa.programs.serde import (
    RankMismatchError,
    expression_from_dict,
    expression_to_dict,
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

    # The organiser's grader raises SyntaxError on `:=`, so the shipped form must not contain
    # one. Repeating the shared work is what that costs: a five-member cohort lands near 167k
    # characters against 6k named, and the cap is what stops the trade at the point where a
    # wider cohort would be better off falling back.
    compiled = compile_expression(program)
    assert ":=" not in compiled
    assert len(compiled) < MAX_QUERY_CHARS

    # The walrus form stays reachable so the scored submissions can still be reproduced.
    named = compile_expression(program, inline=False)
    assert named.startswith("((_v0 :=") and named.endswith(")[-1]")
    assert execute_expression(named, frames) == 200.0
    assert len(named) * 20 < len(compiled)


def test_ranking_without_keys_ranks_the_members_themselves() -> None:
    """Two rules that rejected 54 questions were our refusals, not the model's mistakes.

    `argmax` with no keys is max over the members, and keys handed to an operator that cannot
    rank are surplus rather than contradictory. Neither reading is ambiguous, so neither should
    send a question to the fallback answer.
    """
    frames = _cohort([[30.0], [10.0], [20.0]])
    members = tuple(CellExpr(f"df{index}", 0, 1) for index in range(1, 4))

    assert execute_expression(compile_expression(SelectExpr("argmax", members)), frames) == 30.0
    assert execute_expression(compile_expression(SelectExpr("argmin", members)), frames) == 10.0

    surplus = SelectExpr("sum", members, keys=members)
    assert execute_expression(compile_expression(surplus), frames) == 60.0
    middle = SelectExpr("median", members, keys=members)
    assert execute_expression(compile_expression(middle), frames) == 20.0

    # A key list that does not line up with the members is still a real contradiction.
    with pytest.raises(ValueError, match="one key per member"):
        compile_expression(SelectExpr("argmax", members, keys=members[:2]))


def test_a_doubted_program_gives_way_only_when_nothing_better_is_left() -> None:
    """125 of the 534 fallbacks in the full run were refused by a standard their replacement
    never had to meet.

    The fallback reads a keyword-matched cell out of the same undeclared tables, without a unit
    check, a coverage check or a ranking. So refusing the model's program does not avoid the
    risk it was refused for -- it keeps the risk and discards the reasoning. These checks stay
    strict while the model can still be told to fix them, and give way at the end.
    """
    unlabelled = pd.DataFrame(
        {
            "row_index": [0],
            "column_index": [1],
            "ticker": ["AAA"],
            "report_year": [2024],
            "column_label": ["2024"],
            "source_unit": ["UNKNOWN"],
            "numeric_value": [5.0],
            "base_value": [5.0],
        }
    )
    claimed = CellExpr("df1", 0, 1, dimension="VND")
    strict = {
        "selected_variables": ["df1"],
        "frames": {"df1": unlabelled},
        "target_unit": "VND",
        "target_divisor": 1.0,
    }
    with pytest.raises(SoftGroundingError, match="explicit source-unit lineage"):
        prepare_program(claimed, **strict)  # type: ignore[arg-type]
    prepared, dimension = prepare_program(claimed, **strict, lenient=True)  # type: ignore[arg-type]
    assert dimension == "VND"
    assert execute_expression(compile_expression(prepared), {"df1": unlabelled}) == 5.0

    # A year the program never reads is a doubt about the program, not proof it will not run.
    with pytest.raises(SoftGroundingError, match="required years"):
        validate_query_coverage(
            claimed, frames={"df1": unlabelled}, required_tickers=[], required_years=[2023]
        )
    validate_query_coverage(
        claimed,
        frames={"df1": unlabelled},
        required_tickers=[],
        required_years=[2023],
        lenient=True,
    )

    # A key list that does not line up has a reading that runs: rank the members themselves,
    # which is already what a missing key list means.
    members = [
        {"kind": "cell", "variable": f"df{index}", "row_index": 0, "column_index": 1}
        for index in range(1, 4)
    ]
    ranked = {"kind": "select", "operator": "argmax", "members": members, "keys": members[:2]}
    with pytest.raises(RankMismatchError, match="3 members but 2 keys"):
        expression_from_dict(ranked)
    relaxed = expression_from_dict(ranked, lenient=True)
    assert isinstance(relaxed, SelectExpr)
    assert relaxed.keys is None

    # What must never give way: a program that cannot produce a number at all.
    empty = unlabelled.assign(base_value=[None], numeric_value=[None])
    with pytest.raises(ValueError, match="no numeric value"):
        prepare_program(
            CellExpr("df1", 0, 1),
            selected_variables=["df1"],
            frames={"df1": empty},
            target_unit="VND",
            target_divisor=1.0,
            lenient=True,
        )


def test_program_round_trips_through_serde() -> None:
    """Every node the reader accepts, the writer must produce, and back again unchanged.

    `dataclasses.asdict` was standing in for the writer and looks like it works: it returns a
    nested dict of the right shape with the type of every child silently dropped. Only a
    single-node program survived, which is why the synthetic sampler's four multi-node families
    all wrote a `program` field nothing could read back.
    """
    cells = tuple(CellExpr(f"df{index}", index, 1, dimension="VND") for index in range(1, 4))
    programs: list[object] = [
        cells[0],
        LiteralExpr(2021.0, "YEAR"),
        BinaryExpr("*", BinaryExpr("/", cells[0], cells[1]), LiteralExpr(100.0)),
        AggregateExpr("mean", cells),
        CountIfExpr(cells, ">", LiteralExpr(0.0)),
        ArgExtremumExpr("argmax", keys=cells, values=(LiteralExpr(2021.0, "YEAR"),) * 3),
        SelectExpr("median", cells),
        SelectExpr(
            "argmax",
            cells,
            conditions=(Condition(cells, ">", LiteralExpr(0.0)),),
            keys=cells,
        ),
        SelectExpr(
            "sum",
            cells,
            conditions=(Condition(cells, "<", tuple(LiteralExpr(float(n)) for n in (1, 2, 3))),),
        ),
    ]
    for program in programs:
        payload = expression_to_dict(program)  # type: ignore[arg-type]
        # Through JSON as well: this is written to a file and read back in another process.
        restored = expression_from_dict(json.loads(json.dumps(payload)))
        assert restored == program, payload

    with pytest.raises(TypeError, match="Unsupported expression"):
        expression_to_dict("not a program")  # type: ignore[arg-type]


def test_misaligned_conditions_report_both_counts_so_a_retry_can_fix_them() -> None:
    """A retry is only worth an attempt if the error tells the model what to change.

    Question 473 spent all three of its attempts on the wordless form of this message and came
    back with the same mismatch every time. Both gates have to say it: serde rejects before the
    compiler ever sees the program, so a message improved in one place alone is never read.
    """
    members = [
        {"kind": "cell", "variable": f"df{index}", "row_index": 0, "column_index": 1}
        for index in range(1, 5)
    ]
    node = {
        "kind": "select",
        "operator": "argmax",
        "members": members,
        "keys": members,
        "conditions": [
            {"left": members[:2], "comparator": ">", "right": {"kind": "literal", "value": 0}}
        ],
    }
    with pytest.raises(ValueError, match=r"has 4 members but this condition lists 2 entries"):
        expression_from_dict(node)

    cells = tuple(CellExpr(f"df{index}", 0, 1) for index in range(1, 5))
    program = SelectExpr("argmax", cells, conditions=(Condition(cells[:2], ">", LiteralExpr(0.0)),))
    with pytest.raises(ValueError, match=r"has 4 members but this condition lists 2 entries"):
        compile_expression(program)

    per_member = SelectExpr(
        "argmax",
        cells,
        conditions=(Condition(cells, ">", (LiteralExpr(0.0), LiteralExpr(1.0))),),
    )
    with pytest.raises(ValueError, match=r"4 members but 2 thresholds"):
        compile_expression(per_member)


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


def test_keys_are_relaxed_where_the_model_output_is_parsed_not_only_where_it_compiles() -> None:
    """The compiler's leniency is unreachable if the parser rejects the program first.

    Relaxing the key rules in the compiler alone changed nothing in practice: `expression_from_dict`
    raised on the same shapes, so a program carrying surplus keys never reached the compiler at all.
    Question 473 kept failing on `Keys are only meaningful for argmin and argmax` after the compiler
    was supposedly fixed, which is how this surfaced.
    """
    members = [
        {"kind": "cell", "variable": f"df{index}", "row_index": 0, "column_index": 1}
        for index in (1, 2, 3)
    ]

    ranked = expression_from_dict({"kind": "select", "operator": "argmax", "members": members})
    assert ranked.keys is None
    compile_expression(ranked)

    surplus = expression_from_dict(
        {"kind": "select", "operator": "sum", "members": members, "keys": members}
    )
    assert surplus.keys is None

    with pytest.raises(ValueError, match="one key per member"):
        expression_from_dict(
            {"kind": "select", "operator": "argmax", "members": members, "keys": members[:2]}
        )
