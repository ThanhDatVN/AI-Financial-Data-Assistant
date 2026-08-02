from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

merger = importlib.import_module("scripts.34_merge_retrieval_shards")


def _write_shard(path: Path, *, index: int, question_id: int) -> None:
    path.mkdir()
    metadata = {
        "retrieval_sha256": "a" * 64,
        "records_sha256": "b" * 64,
        "model": "reranker",
        "shard_count": 2,
        "shard_index": index,
    }
    (path / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    row = {
        "id": question_id,
        "question": "q",
        "reranked": [f"table_{question_id}"],
        "fused": [f"table_{question_id}"],
    }
    (path / "retrieval.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_merge_retrieval_shards_validates_and_orders_rows(tmp_path: Path, monkeypatch: Any) -> None:
    shard_0 = tmp_path / "shard_0"
    shard_1 = tmp_path / "shard_1"
    _write_shard(shard_0, index=0, question_id=2)
    _write_shard(shard_1, index=1, question_id=1)
    output = tmp_path / "merged.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "34_merge_retrieval_shards.py",
            str(shard_0),
            str(shard_1),
            "--output",
            str(output),
            "--expected-rows",
            "2",
        ],
    )
    merger.main()

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["id"] for row in rows] == [1, 2]
    metadata = json.loads(output.with_suffix(".jsonl.metadata.json").read_text())
    assert metadata["merged_shards"] == 2
