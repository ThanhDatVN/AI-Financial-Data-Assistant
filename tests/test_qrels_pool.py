from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import jsonschema

pooler = import_module("scripts.61_build_qrels_pool")


def test_pool_union_preserves_provenance_and_is_deterministic() -> None:
    runs = [
        (
            "sparse",
            [
                {
                    "id": 7,
                    "question": "q",
                    "bm25": ["doc|table_2", "doc|table_1", "doc|table_2"],
                    "fused": ["doc|table_2"],
                }
            ],
        ),
        (
            "dense",
            [
                {
                    "id": 7,
                    "question": "q",
                    "dense": ["doc|table_1", "doc|table_3"],
                }
            ],
        ),
    ]
    first = pooler._merge_rankings(runs, selected_ids={7}, depth=2)
    second = pooler._merge_rankings(runs, selected_ids={7}, depth=2)

    assert first == second
    candidates = first[7]["candidates"]
    assert [item["table_ref"] for item in candidates] == [
        "doc|table_2",
        "doc|table_1",
        "doc|table_3",
    ]
    assert len(candidates[0]["sources"]) == 2
    assert len(candidates[1]["sources"]) == 2
    assert all(item["relevance"] == "unjudged" for item in candidates)


def test_pool_respects_selected_ids_and_depth() -> None:
    runs = [
        (
            "r",
            [
                {"id": 1, "question": "one", "bm25": ["a", "b"]},
                {"id": 2, "question": "two", "bm25": ["c"]},
            ],
        )
    ]
    result = pooler._merge_rankings(runs, selected_ids={1}, depth=1)
    assert set(result) == {1}
    assert [item["table_ref"] for item in result[1]["candidates"]] == ["a"]


def test_checked_in_pool_matches_schema_and_has_no_duplicate_refs() -> None:
    root = Path(__file__).resolve().parents[1]
    schema = json.loads((root / "configs/qrels_pool.schema.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (root / "annotations/qrels_pool.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 100
    for row in rows:
        jsonschema.validate(row, schema)
        refs = [candidate["table_ref"] for candidate in row["candidates"]]
        assert refs
        assert len(refs) == len(set(refs))
