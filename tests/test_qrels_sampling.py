from __future__ import annotations

import hashlib
import json
from importlib import import_module
from pathlib import Path

import jsonschema

sampler = import_module("scripts.60_sample_qrels")


def test_round_robin_sampling_is_deterministic_and_covers_groups() -> None:
    groups = {
        "a": [{"id": 1}, {"id": 2}],
        "b": [{"id": 3}, {"id": 4}],
        "c": [{"id": 5}],
    }
    first = sampler._sample_round_robin(groups, size=3, seed=7)
    second = sampler._sample_round_robin(groups, size=3, seed=7)
    assert first == second
    assert {int(row["id"]) for row in first} & {1, 2}
    assert {int(row["id"]) for row in first} & {3, 4}
    assert {int(row["id"]) for row in first} & {5}


def test_checked_in_qrels_template_matches_v2_schema() -> None:
    root = Path(__file__).resolve().parents[1]
    schema = json.loads((root / "configs/qrels.schema.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (root / "annotations/qrels_template.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(rows) == 100
    assert len({row["id"] for row in rows}) == 100
    for row in rows:
        jsonschema.validate(row, schema)
        expected = hashlib.sha256(sampler.ascii_words(row["question"]).encode("utf-8")).hexdigest()
        assert row["split_group"]["question_fingerprint"] == expected
