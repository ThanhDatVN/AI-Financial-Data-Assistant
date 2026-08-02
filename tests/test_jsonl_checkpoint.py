from __future__ import annotations

import json
from pathlib import Path

import pytest

from vifinqa.checkpoints.jsonl import JsonlRowCheckpoint


def test_jsonl_row_checkpoint_is_atomic_deterministic_and_resumable(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint"
    checkpoint = JsonlRowCheckpoint(path, fingerprint={"run": "a"})
    checkpoint.write({"id": 2, "value": "two"})
    checkpoint.write({"id": 1, "value": "one"})
    checkpoint.write({"id": 1, "value": "one"})

    resumed = JsonlRowCheckpoint(path, fingerprint={"run": "a"})
    assert resumed.completed_ids() == {1, 2}
    output = tmp_path / "output.jsonl"
    rows = resumed.consolidate(output)
    assert [row["id"] for row in rows] == [1, 2]
    assert [json.loads(line)["id"] for line in output.read_text().splitlines()] == [1, 2]
    assert not list(tmp_path.rglob("*.tmp"))


def test_jsonl_row_checkpoint_rejects_mismatched_runs_and_rows(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint"
    checkpoint = JsonlRowCheckpoint(path, fingerprint={"run": "a"})
    checkpoint.write({"id": 1, "value": "one"})
    with pytest.raises(ValueError, match="different run fingerprint"):
        JsonlRowCheckpoint(path, fingerprint={"run": "b"})
    with pytest.raises(ValueError, match="different content"):
        checkpoint.write({"id": 1, "value": "changed"})
