from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import pandas as pd

from vifinqa.programs.grounding import cells_in_program, referenced_variables
from vifinqa.programs.serde import expression_from_dict

runner = import_module("scripts.50_generate_programs")


def test_generation_fingerprint_detects_model_and_input_changes(tmp_path: Path) -> None:
    retrieval = tmp_path / "retrieval.jsonl"
    manifest = tmp_path / "manifest.parquet"
    retrieval.write_text(json.dumps({"id": 1}) + "\n", encoding="utf-8")
    manifest.write_bytes(b"manifest-v1")

    first = runner._fingerprint(
        retrieval=retrieval,
        manifest=manifest,
        model="open/model",
        model_revision="abc123",
        candidate_tables=10,
        max_tokens=1024,
        context_limit=16384,
        execution_timeout=10.0,
        request_timeout=180.0,
        memory_limit_mb=None,
        thinking_mode="disabled",
        max_attempts=3,
    )
    second = runner._fingerprint(
        retrieval=retrieval,
        manifest=manifest,
        model="open/model",
        model_revision="def456",
        candidate_tables=10,
        max_tokens=1024,
        context_limit=16384,
        execution_timeout=10.0,
        request_timeout=180.0,
        memory_limit_mb=None,
        thinking_mode="disabled",
        max_attempts=3,
    )
    assert first != second
    assert first["retrieval_sha256"] == runner._sha256(retrieval)
    assert first["thinking_mode"] == "disabled"
    assert first["max_attempts"] == 3
    assert first["semantic_convention_version"] == 1
    assert first["table_unit_inference_version"] == 3
    assert first["question_count"] == 0
    assert first["table_unit_source"] == "latest"
    assert first["project_revision"] is None

    pinned_project = runner._fingerprint(
        retrieval=retrieval,
        manifest=manifest,
        model="open/model",
        model_revision="abc123",
        candidate_tables=10,
        max_tokens=1024,
        context_limit=16384,
        execution_timeout=10.0,
        request_timeout=180.0,
        memory_limit_mb=None,
        thinking_mode="disabled",
        max_attempts=3,
        project_revision="1" * 40,
    )
    assert pinned_project != first

    retrieval.write_text(json.dumps({"id": 2}) + "\n", encoding="utf-8")
    third = runner._fingerprint(
        retrieval=retrieval,
        manifest=manifest,
        model="open/model",
        model_revision="abc123",
        candidate_tables=10,
        max_tokens=1024,
        context_limit=16384,
        execution_timeout=10.0,
        request_timeout=180.0,
        memory_limit_mb=None,
        thinking_mode="disabled",
        max_attempts=3,
    )
    assert third != first

    thinking = runner._fingerprint(
        retrieval=retrieval,
        manifest=manifest,
        model="open/model",
        model_revision="abc123",
        candidate_tables=10,
        max_tokens=1024,
        context_limit=16384,
        execution_timeout=10.0,
        request_timeout=180.0,
        memory_limit_mb=None,
        thinking_mode="auto",
        max_attempts=3,
    )
    assert thinking != third

    hierarchy = runner._fingerprint(
        retrieval=retrieval,
        manifest=manifest,
        model="open/model",
        model_revision="abc123",
        candidate_tables=10,
        max_tokens=1024,
        context_limit=16384,
        execution_timeout=10.0,
        request_timeout=180.0,
        memory_limit_mb=None,
        thinking_mode="disabled",
        max_attempts=3,
        row_hierarchy=True,
    )
    assert hierarchy != third
    assert hierarchy["row_hierarchy"] is True

    selected = runner._fingerprint(
        retrieval=retrieval,
        manifest=manifest,
        model="open/model",
        model_revision="abc123",
        candidate_tables=10,
        max_tokens=1024,
        context_limit=16384,
        execution_timeout=10.0,
        request_timeout=180.0,
        memory_limit_mb=None,
        thinking_mode="disabled",
        max_attempts=3,
        selected_question_ids=[5, 9, 12],
    )
    reordered = runner._fingerprint(
        retrieval=retrieval,
        manifest=manifest,
        model="open/model",
        model_revision="abc123",
        candidate_tables=10,
        max_tokens=1024,
        context_limit=16384,
        execution_timeout=10.0,
        request_timeout=180.0,
        memory_limit_mb=None,
        thinking_mode="disabled",
        max_attempts=3,
        selected_question_ids=[12, 9, 5],
    )
    assert selected["question_count"] == 3
    assert selected["question_ids_sha256"] != reordered["question_ids_sha256"]

    manifest_units = runner._fingerprint(
        retrieval=retrieval,
        manifest=manifest,
        model="open/model",
        model_revision="abc123",
        candidate_tables=10,
        max_tokens=1024,
        context_limit=16384,
        execution_timeout=10.0,
        request_timeout=180.0,
        memory_limit_mb=None,
        thinking_mode="disabled",
        max_attempts=3,
        selected_question_ids=[5, 9, 12],
        table_unit_source="manifest",
    )
    assert manifest_units != selected


def test_candidate_limit_expands_to_entity_year_routes() -> None:
    row: dict[str, object] = {
        "query_spec": {"tickers": ["AAA", "BBB", "CCC"], "years": [2023, 2024]}
    }
    assert runner._candidate_limit(row, minimum=4) == 6
    assert runner._candidate_limit(row, minimum=10) == 10


def test_selected_variables_come_from_the_program_not_the_declaration() -> None:
    # The model kept declaring every table it had looked at, which cost whole questions to
    # bookkeeping. The tree it emitted is the only authority on what the program reads.
    program = {
        "kind": "binary",
        "operator": "-",
        "left": {
            "kind": "cell",
            "variable": "df6",
            "row_index": 0,
            "column_index": 1,
            "value_column": "base_value",
            "dimension": "VND",
        },
        "right": {
            "kind": "cell",
            "variable": "df1",
            "row_index": 2,
            "column_index": 1,
            "value_column": "base_value",
            "dimension": "VND",
        },
    }
    expression = expression_from_dict(program)
    assert sorted(referenced_variables(expression)) == ["df1", "df6"]
    assert len(cells_in_program(expression)) == 2


def test_fallback_keeps_an_unsolved_question_in_the_submission() -> None:
    # The organiser discards a submission with any question missing, so a question the model
    # never solved costs the whole run rather than its own points.
    frame = pd.DataFrame(
        {
            "row_index": [0, 1, 1],
            "column_index": [0, 1, 2],
            "source_unit": ["MILLION_VND"] * 3,
            "numeric_value": [None, 7.0, 9.0],
            "base_value": [None, 7_000_000.0, 9_000_000.0],
        }
    )
    frame["row_label"] = ["Tổng cộng", "Lãi tiền gửi", "Lãi tiền gửi"]
    frame["column_label"] = ["2017", "2018", "2017"]

    fallback = runner._fallback_program(
        frames={"df1": frame},
        target_divisor=1_000_000.0,
        question="Lãi tiền gửi năm 2018 là bao nhiêu triệu đồng?",
        years=[2018],
    )
    assert fallback is not None
    query, selected, answer = fallback
    assert selected == ["df1"]
    # The question names the row and the year, so the fallback aims at that figure rather
    # than at whichever cell happens to come first.
    assert answer == 7.0
    assert "base_value" in query and "df1" in query

    blank = pd.DataFrame(
        {
            "row_index": [0],
            "column_index": [0],
            "source_unit": ["UNKNOWN"],
            "numeric_value": [None],
            "base_value": [None],
        }
    )
    assert runner._fallback_program(frames={"df1": blank}, target_divisor=1.0, question="x") is None


def test_select_rows_uses_stable_requested_ids_without_changing_run_identity() -> None:
    rows = [{"id": 1}, {"id": "2"}, {"id": 3}]
    assert runner._select_rows(rows, question_ids=[3, 1], limit=None) == [rows[2], rows[0]]
    assert runner._select_rows(rows, question_ids=[3, 1], limit=1) == [rows[2]]


def test_select_rows_partitions_into_stable_disjoint_shards() -> None:
    rows = [{"id": index} for index in range(1, 8)]
    shards = [
        runner._select_rows(
            rows,
            question_ids=None,
            limit=None,
            shard_count=2,
            shard_index=index,
        )
        for index in range(2)
    ]
    assert shards[0] == rows[::2]
    assert shards[1] == rows[1::2]
    assert {int(row["id"]) for shard in shards for row in shard} == set(range(1, 8))


def test_select_rows_rejects_duplicate_and_missing_ids() -> None:
    rows = [{"id": 1}, {"id": 2}]
    for question_ids in ([1, 1], [3]):
        try:
            runner._select_rows(rows, question_ids=question_ids, limit=None)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid IDs to fail: {question_ids}")

    try:
        runner._select_rows([{"id": 1}, {"id": "1"}], question_ids=None, limit=None)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected duplicate retrieval IDs to fail")


def test_a_session_cap_does_not_change_what_the_run_is_about() -> None:
    """A capped first session must hand its checkpoint to an uncapped second one.

    Kaggle keeps nothing from a session it kills, so a long run is split: notebook 02 answers a
    fixed slice and saves, notebook 03 finishes the rest. That only works if both agree on the
    run's identity. Folding `--limit` into it meant the capped session wrote a fingerprint the
    uncapped one could never match, and the resume refused the checkpoint it was handed.
    """
    rows = [{"id": index, "fused": []} for index in range(1, 21)]

    scope = runner._select_rows(rows, question_ids=None, limit=None)
    capped = scope[:12]
    assert [row["id"] for row in capped] == list(range(1, 13))

    # The cap changes what this session works through, not which questions the run covers.
    assert [row["id"] for row in scope] == list(range(1, 21))

    # And an explicit selection still narrows the run itself, because that is the run's identity.
    narrowed = runner._select_rows(rows, question_ids=[3, 1, 2], limit=None)
    assert [row["id"] for row in narrowed] == [3, 1, 2]


def test_sharding_sends_a_question_to_the_same_shard_in_both_sessions() -> None:
    """Position-based sharding has to survive the second session widening the slice.

    Notebook 02 shards the first 600 of 1,012; notebook 03 shards all 1,012. A question that
    moved shards between the two would be answered twice or not at all.
    """
    rows = [{"id": index, "fused": []} for index in range(1, 101)]
    shard_count = 8

    def shard_of(subset: list[dict[str, object]], question_id: int) -> int | None:
        for index in range(shard_count):
            assigned = runner._select_rows(
                subset, question_ids=None, limit=None, shard_count=shard_count, shard_index=index
            )
            if any(row["id"] == question_id for row in assigned):
                return index
        return None

    first_session = rows[:60]
    for question_id in (1, 17, 40, 60):
        assert shard_of(first_session, question_id) == shard_of(rows, question_id)


def test_a_context_refusal_is_arithmetic_not_a_bad_program() -> None:
    """A fixed token budget is wrong in both directions on the same run.

    Question 213 carries a 12,289-token prompt, so asking for 4,096 output tokens puts the
    request one token past the 16,384 context and the server refuses before the model writes
    anything -- a question that had been answered correctly turned into a fallback. Question 442
    truncates at the same 4,096 because its prompt is short enough to have afforded far more.
    The refusal states both figures, so the budget can be corrected from it rather than guessed.
    """

    def refusal(requested: int, prompt: int, limit: int = 16384) -> str:
        return (
            f"Error code: 400 - {{'error': {{'message': \"This model's maximum context length is "
            f"{limit} tokens. However, you requested {requested} output tokens and your prompt "
            f"contains at least {prompt} input tokens, for a total of at least "
            f'{requested + prompt} tokens. (parameter=input_tokens, value={prompt})"}}}}'
        )

    budget = runner._TokenBudget(4096, 16384)
    assert budget.current == 4096
    assert budget.shrink(refusal(4096, 12354))
    # The refusal reports the prompt as "at least" that long, so shaving the arithmetic exactly
    # landed one token short and bought a second refusal -- question 213 spent every correction
    # measuring its own prompt and completed no call at all. Each refusal now doubles the cushion.
    first = budget.current
    assert first == 16384 - 12354 - 128
    assert budget.shrink(refusal(first, 12419))
    assert budget.current == 16384 - 12419 - 256
    assert budget.current + 12419 < 16384

    # Anything else is a real bad request and must keep propagating.
    assert not runner._TokenBudget(4096, 16384).shrink("Error code: 400 - malformed schema")

    # A prompt that leaves no useful room is not worth retrying either.
    crowded = "maximum context length is 16384 tokens ... (parameter=input_tokens, value=16300)"
    assert not runner._TokenBudget(4096, 16384).shrink(crowded)


def test_a_truncated_program_is_unfinished_rather_than_wrong() -> None:
    """Question 442 stopped at exactly 4,096 tokens on all three attempts and never parsed.

    Retrying an unfinished program at the budget that cut it off reproduces the same cut in the
    same place. Its prompt was short enough to afford far more, and the only figure that says so
    is `usage.prompt_tokens`, which a truncation -- unlike a refusal -- never reports on its own.
    """
    budget = runner._TokenBudget(4096, 16384)
    assert budget.widen(6000)
    assert budget.current == 16384 - 6000 - 64
    assert budget.current > 4096

    # A prompt that already fills the context has nothing left to give, and the caller must let
    # the failure stand rather than loop.
    assert not runner._TokenBudget(4096, 16384).widen(16000)
    assert not runner._TokenBudget(4096, 16384).widen(None)

    # Observing a long prompt lowers the budget before the server has to refuse it at all --
    # question 213's failure mode, prevented rather than corrected.
    measured = runner._TokenBudget(4096, 16384)
    measured.observe(12419)
    assert measured.current == 16384 - 12419 - 64
    # Measuring only ever lowers. Were it allowed to raise, the response that follows a widened
    # retry would clamp the budget back to the ceiling that truncated the program to begin with,
    # and the retry would stop in the same place for the same reason.
    measured.observe(1000)
    assert measured.current == 16384 - 12419 - 64

    widened = runner._TokenBudget(4096, 16384)
    assert widened.widen(6000)
    widened.observe(6000)
    assert widened.current == 16384 - 6000 - 64
