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
    )
    assert first != second
    assert first["retrieval_sha256"] == runner._sha256(retrieval)

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
    )
    assert third != first


def test_candidate_limit_expands_to_entity_year_routes() -> None:
    row: dict[str, object] = {
        "query_spec": {"tickers": ["AAA", "BBB", "CCC"], "years": [2023, 2024]}
    }
    assert runner._candidate_limit(row, minimum=4) == 6
    assert runner._candidate_limit(row, minimum=10) == 10
