from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import pandas as pd

from vifinqa.evidence.store import parsed_table_to_long_frame
from vifinqa.generation.prompt import CandidateSchema, numeric_cells_of
from vifinqa.indexing.manifest import ManifestRecord
from vifinqa.parsing.models import RawTable
from vifinqa.parsing.table_parser import parse_table
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


def test_the_prompt_is_measured_rather_than_inferred_from_a_refusal() -> None:
    """Two unrelated questions died asking 5,245 tokens of an 11,140-token prompt in a 16,384
    context -- identical figures, which no reading of the correction arithmetic explained.

    The prompt grows between attempts because each retry carries the previous failure's feedback,
    so a budget derived from the refusal is already stale when it is used. Measuring the exact
    messages before every attempt is the only version of this that cannot drift.
    """
    assert runner._tokenize_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/tokenize"
    assert runner._tokenize_url("http://127.0.0.1:8000/v1/") == "http://127.0.0.1:8000/tokenize"
    assert runner._tokenize_url("http://host:9/") == "http://host:9/tokenize"

    # A server without the route must not take the run down with it; the refusal path still
    # catches the overflow, just one request later.
    assert runner._measure_prompt("http://127.0.0.1:1/tokenize", "m", [], 1.0) == 0

    # Feedback lengthens the prompt, and the budget has to follow it down rather than hold the
    # figure it read before the feedback existed.
    budget = runner._TokenBudget(6144, 16384)
    budget.observe(11140)
    first = budget.current
    assert first == 16384 - 11140 - 64
    budget.observe(11270)  # the same prompt plus a retry's feedback
    assert budget.current == 16384 - 11270 - 64
    assert budget.current < first
    assert budget.current + 11270 < 16384


def _schema(variable: str) -> CandidateSchema:
    raw = RawTable(
        1,
        1,
        1,
        0,
        (
            "<table><tr><td>Chỉ tiêu</td><td>2024</td></tr>"
            "<tr><td>Doanh thu</td><td>12</td></tr></table>"
        ),
        ("Đơn vị: VND",),
        None,
    )
    table = parse_table(raw)
    record = ManifestRecord(
        f"DOC_{variable}|table_1",
        f"DOC_{variable}",
        "AAA",
        2024,
        "consolidated",
        1,
        1,
        1,
        0,
        None,
        "VND",
        1,
        2,
        2,
        table.headers,
        ("Doanh thu",),
        "",
        Path("report.txt").as_posix(),
        "0" * 64,
    )
    frame = parsed_table_to_long_frame(record, table)
    return CandidateSchema(variable, record, table, numeric_cells_of(frame))


def test_a_prompt_with_no_room_behind_it_gives_up_candidates_not_the_question() -> None:
    """Measuring a prompt exactly does not help when no budget fits behind it either way.

    The tables at the bottom of the ranking are the least likely to hold the answer, so they are
    what gives. Without a tokenizer to measure against, nothing is dropped -- guessing is what
    the measurement replaced, and it must not come back through this door.
    """
    schemas = [_schema(f"df{index}") for index in range(1, 21)]
    lengths = {20: 15_500, 19: 14_800, 18: 14_100, 17: 13_000}

    def fake_measure(url: str, model: str, messages: list[dict[str, str]], timeout: float) -> int:
        rendered = messages[1]["content"]
        shown = sum(1 for schema in schemas if f'variable="{schema.variable}"' in rendered)
        return lengths.get(shown, 9_000)

    fitted = runner._fit_prompt(
        question="q",
        schemas=schemas,
        target_unit="VND",
        target_divisor=1.0,
        required_tickers=[],
        required_years=[],
        row_hierarchy=False,
        tokenize_url="http://unused/tokenize",
        model="m",
        context_limit=16_384,
        max_tokens=6_144,
        request_timeout=60.0,
        _measure=fake_measure,
    )
    # 16,384 - 14,100 - 64 = 2,220, the first that clears the 1,536 a cohort program needs.
    assert len(fitted[2]) == 18

    unmeasurable = runner._fit_prompt(
        question="q",
        schemas=schemas,
        target_unit="VND",
        target_divisor=1.0,
        required_tickers=[],
        required_years=[],
        row_hierarchy=False,
        tokenize_url="http://127.0.0.1:1/tokenize",
        model="m",
        context_limit=16_384,
        max_tokens=6_144,
        request_timeout=1.0,
    )
    assert len(unmeasurable[2]) == 20


def test_the_budget_never_asks_for_more_than_the_timeout_can_decode() -> None:
    """Context is not the only ceiling, and widening into the other one only buys a timeout.

    The 8B decoded 4,096 tokens in 171 seconds against a 360-second request timeout. The 14B is
    roughly half that rate, so a budget the 16,384 context happily affords can cost more wall
    clock than the request is allowed to take. The rate is measured per question because it has
    differed by 6.6x between quantisation kernels alone.
    """
    budget = runner._TokenBudget(6144, 16384)
    # 4,096 tokens in 171 seconds is about 24/s; four fifths of 360 seconds buys about 6,900.
    budget.observe_rate(4096, 171.0, 360.0)
    assert budget.affordable is not None
    assert 6500 < budget.affordable < 7200
    assert budget.current == 6144  # the configured ceiling is still lower, so it stands

    # A short prompt leaves 10k of context, but the clock does not, and widening must respect it.
    assert budget.widen(6000)
    assert budget.current == budget.affordable
    assert budget.current < 16384 - 6000 - 64

    # Half the rate halves the budget, and the ceiling that was safe for the 8B is not for a 14B.
    slower = runner._TokenBudget(6144, 16384)
    slower.observe_rate(4096, 342.0, 360.0)
    assert slower.current < 4096

    # Nothing to divide by is not a measurement.
    unmeasured = runner._TokenBudget(6144, 16384)
    unmeasured.observe_rate(None, 171.0, 360.0)
    unmeasured.observe_rate(4096, 0.0, 360.0)
    assert unmeasured.affordable is None
    assert unmeasured.current == 6144


def test_the_citation_probe_shares_the_generator_s_candidate_rule() -> None:
    """The probe measures the set the generator built, so it must count it the same way.

    Citing the top 20 of the ranking would be a different set: the generator skips candidates
    whose table parses to no number, and allows a cohort question one candidate per route. Both
    make the real set wider, and understating it would understate the very quantity the
    submission is spent to measure.
    """
    probe = import_module("scripts.46_cite_prompt_candidates")
    for tickers, years in (([], []), (["AAA"], [2024]), (["A", "B", "C"], [2023, 2024])):
        spec = {"tickers": tickers, "years": years}
        row = {"query_spec": spec}
        assert probe._candidate_limit(spec, minimum=20) == runner._candidate_limit(row, minimum=20)
    # A cohort question is allowed one candidate per route once that exceeds the floor.
    assert probe._candidate_limit({"tickers": ["A"] * 6, "years": [1, 2, 3]}, minimum=10) == 18
    assert probe._candidate_limit({"tickers": ["A"] * 6, "years": [1, 2, 3]}, minimum=20) == 20


def test_the_scope_router_changes_the_prompt_and_the_fingerprint(tmp_path: Path) -> None:
    """Two policies show two different prompts, so a resume must not join them silently."""

    class _Store:
        def __init__(self, records: dict[str, ManifestRecord]) -> None:
            self.records = records

    def _record(scope: str, table: int) -> ManifestRecord:
        doc = f"AAA_financial_statements_2024_{scope}"
        return ManifestRecord(
            table_ref=f"{doc}|table_{table}",
            doc_id=doc,
            ticker="AAA",
            report_year=2024,
            scope=scope,
            table_id=table,
            page_no=1,
            line_no=table * 10,
            char_offset=0,
            section_title=None,
            unit="VND",
            header_rows=1,
            n_rows=2,
            n_cols=2,
            headers=("Chỉ tiêu", "2024"),
            row_labels=("Doanh thu thuần",),
            retrieval_text="Doanh thu thuần",
            source_path="report.txt",
            html_sha256="0" * 64,
        )

    separate = _record("separate", 1)
    consolidated = _record("consolidated", 2)
    store = _Store({record.table_ref: record for record in (separate, consolidated)})
    row = {
        "id": 1,
        "fused": [separate.table_ref, consolidated.table_ref],
        "query_spec": {"tickers": ["AAA"], "years": [2024], "scope": None},
    }
    assert runner._routed_refs(row, store=store, policy="both") == [
        separate.table_ref,
        consolidated.table_ref,
    ]
    assert runner._routed_refs(row, store=store, policy="consolidated") == [consolidated.table_ref]
    # The failure path cites the candidates a lost question was shown, so it has to survive a row
    # that never got far enough to be well formed.
    assert runner._routed_refs({"id": 1}, store=store, policy="consolidated") == []

    retrieval = tmp_path / "retrieval.jsonl"
    manifest = tmp_path / "manifest.parquet"
    retrieval.write_text(json.dumps(row) + "\n", encoding="utf-8")
    manifest.write_bytes(b"manifest-v1")
    fingerprints = [
        runner._fingerprint(
            retrieval=retrieval,
            manifest=manifest,
            model="open/model",
            model_revision="abc123",
            candidate_tables=20,
            max_tokens=1024,
            context_limit=16384,
            execution_timeout=10.0,
            request_timeout=180.0,
            memory_limit_mb=None,
            thinking_mode="disabled",
            max_attempts=3,
            scope_router=policy,
        )
        for policy in ("both", "consolidated")
    ]
    assert fingerprints[0] != fingerprints[1]
    assert fingerprints[0]["scope_router"] == "both"


def test_the_citation_probe_routes_scope_the_way_the_generator_does() -> None:
    """One rule, imported by both, because a probe that measures a different set measures nothing.

    The probe's whole job is to report the recall of the set the generator will build. If it
    routed scope by its own copy of the rule, a submission spent on the question would answer
    about a list nothing ever showed the model.
    """
    probe = import_module("scripts.46_cite_prompt_candidates")
    assert probe.scope_routed is runner.scope_routed
    body = Path(probe.__file__).read_text(encoding="utf-8")
    assert "policy=args.scope_router" in body


def test_the_citation_probe_leaves_the_answer_alone(tmp_path: Path) -> None:
    """Only the two citation lists may change, or the experiment costs a submission and a run.

    EXECUTION and ANSWER have to come back exactly as they scored, because the point is to read
    one number off the same run rather than to test a new one.
    """
    probe = import_module("scripts.46_cite_prompt_candidates")
    source = (tmp_path / "46_cite_prompt_candidates.py").write_text(
        Path(probe.__file__).read_text(encoding="utf-8"), encoding="utf-8"
    )
    assert source
    body = Path(probe.__file__).read_text(encoding="utf-8")
    for field in ("answer", "pandas_query", "evidence"):
        assert f'prediction["{field}"] =' not in body, f"the probe must not rewrite {field}"
    assert 'prediction["relevant_tables"] =' in body
    # Documents are left alone by default. The run being repackaged names them from the
    # question's own metadata and scores 0.9537 doing it; moving two numbers at once would
    # trade a known-good one for a second unknown, and only one of them is being measured.
    assert "if args.docs_from_tables:" in body


def test_the_citation_probe_says_its_output_needs_retargeting() -> None:
    """Submissions 3132 and 3133 were spent on a conclusion the project already held.

    The probe writes the manifest's own references, `{doc}|table_N`. The grader reads
    `{doc}|{line_no}` -- settled over five submissions and recorded with "do not re-derive this".
    Both packages came back with every table metric at 0.0 while EXECUTION and ANSWER were
    untouched, which is what an unreadable citation list looks like rather than a low-recall one.

    The note belongs here rather than only in a document read at the start of a session, because
    this is where the references are made.
    """
    probe = import_module("scripts.46_cite_prompt_candidates")
    doc = probe.__doc__ or ""
    assert "42_retarget_table_refs.py --grammar line" in doc
    assert "{doc}|{line_no}" in doc
