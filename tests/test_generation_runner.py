from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

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
        execution_timeout=10.0,
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
        execution_timeout=10.0,
        memory_limit_mb=None,
        thinking_mode="disabled",
        max_attempts=3,
    )
    assert first != second
    assert first["retrieval_sha256"] == runner._sha256(retrieval)
    assert first["thinking_mode"] == "disabled"
    assert first["max_attempts"] == 3
    assert first["project_revision"] is None

    pinned_project = runner._fingerprint(
        retrieval=retrieval,
        manifest=manifest,
        model="open/model",
        model_revision="abc123",
        candidate_tables=10,
        max_tokens=1024,
        execution_timeout=10.0,
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
        execution_timeout=10.0,
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
        execution_timeout=10.0,
        memory_limit_mb=None,
        thinking_mode="auto",
        max_attempts=3,
    )
    assert thinking != third


def test_candidate_limit_expands_to_entity_year_routes() -> None:
    row: dict[str, object] = {
        "query_spec": {"tickers": ["AAA", "BBB", "CCC"], "years": [2023, 2024]}
    }
    assert runner._candidate_limit(row, minimum=4) == 6
    assert runner._candidate_limit(row, minimum=10) == 10


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
