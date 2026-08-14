from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import pytest

compare = import_module("scripts.54_compare_generation_variants")


def _write_root(root: Path, rows: list[dict[str, object]], *, unit_source: str) -> None:
    root.mkdir()
    (root / "submission.json").write_text(json.dumps(rows), encoding="utf-8")
    (root / "run_metadata.json").write_text(
        json.dumps({"table_unit_source": unit_source}), encoding="utf-8"
    )
    traces = [
        {"id": row["id"], "fallback": False, "latency_seconds": 2.0, "completion_tokens": 10}
        for row in rows
    ]
    (root / "program_traces.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in traces), encoding="utf-8"
    )


def test_compare_roots_reports_only_real_field_changes(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    common = {
        "pandas_query": "1.0",
        "evidence": [],
        "relevant_tables": ["a|1"],
        "relevant_docs": ["a"],
    }
    _write_root(
        left,
        [{"id": 1, "answer": 1.0, **common}, {"id": 2, "answer": 2.0, **common}],
        unit_source="manifest",
    )
    _write_root(
        right,
        [
            {"id": 1, "answer": 1.0, **common},
            {"id": 2, "answer": 3.0, **common, "pandas_query": "3.0"},
        ],
        unit_source="latest",
    )

    report = compare.compare_roots(left, right)

    assert report["questions"] == 2
    assert report["left"]["table_unit_source"] == "manifest"
    assert report["right"]["table_unit_source"] == "latest"
    assert report["differences"]["answers"] == 1
    assert report["differences"]["pandas_queries"] == 1
    assert report["differences"]["any_ids"] == [2]


def test_compare_roots_rejects_unmatched_ids(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_root(left, [{"id": 1, "answer": 1.0}], unit_source="manifest")
    _write_root(right, [{"id": 2, "answer": 2.0}], unit_source="latest")

    with pytest.raises(ValueError, match="Variant ID mismatch"):
        compare.compare_roots(left, right)
