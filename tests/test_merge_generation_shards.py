from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import pytest

merger = import_module("scripts.51_merge_generation_shards")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _shard(root: Path, index: int, prediction_id: int) -> Path:
    root.mkdir(parents=True)
    (root / "run_metadata.json").write_text(
        json.dumps(
            {
                "model": "Qwen/Qwen3-8B-AWQ",
                "retrieval_sha256": "a" * 64,
                "shard_count": 2,
                "shard_index": index,
            }
        ),
        encoding="utf-8",
    )
    prediction = {"id": prediction_id, "answer": prediction_id}
    _write_jsonl(root / "predictions.jsonl", [prediction])
    _write_jsonl(root / "program_traces.jsonl", [{"id": prediction_id, "program": "x"}])
    _write_jsonl(root / "errors.jsonl", [])
    _write_jsonl(root / "error_attempts.jsonl", [])
    data = root / "data"
    data.mkdir()
    (data / f"{prediction_id}.csv").write_text("value\n1\n", encoding="utf-8")
    return root


def test_merge_generation_shards_validates_and_orders_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shard_0 = _shard(tmp_path / "shard_0", 0, 3)
    shard_1 = _shard(tmp_path / "shard_1", 1, 2)
    output = tmp_path / "merged"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "51_merge_generation_shards.py",
            str(shard_0),
            str(shard_1),
            "--output",
            str(output),
            "--expected-rows",
            "2",
        ],
    )

    merger.main()

    submission = json.loads((output / "submission.json").read_text(encoding="utf-8"))
    assert [row["id"] for row in submission] == [2, 3]
    assert sorted(path.name for path in (output / "data").glob("*.csv")) == ["2.csv", "3.csv"]
    metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["merged_shards"] == 2


def test_merge_generation_shards_rejects_duplicate_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shard_0 = _shard(tmp_path / "shard_0", 0, 1)
    shard_1 = _shard(tmp_path / "shard_1", 1, 1)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "51_merge_generation_shards.py",
            str(shard_0),
            str(shard_1),
            "--output",
            str(tmp_path / "merged"),
        ],
    )
    with pytest.raises(ValueError, match="Duplicate IDs"):
        merger.main()
