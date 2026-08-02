from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json_atomic(path: Path, payload: object) -> None:
    """Write one JSON value without leaving a valid-looking partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


class JsonlRowCheckpoint:
    """Persist independent JSON rows atomically and consolidate them deterministically."""

    def __init__(
        self,
        path: Path,
        *,
        fingerprint: dict[str, object],
        id_field: str = "id",
    ) -> None:
        self.path = path
        self.id_field = id_field
        self.rows_path = path / "rows"
        self.metadata_path = path / "run_metadata.json"
        self.rows_path.mkdir(parents=True, exist_ok=True)
        if self.metadata_path.exists():
            actual = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if actual != fingerprint:
                raise ValueError("Checkpoint belongs to a different run fingerprint")
        else:
            write_json_atomic(self.metadata_path, fingerprint)

    def _id(self, row: dict[str, object]) -> int:
        value = row.get(self.id_field)
        if isinstance(value, bool) or not isinstance(value, int | str):
            raise TypeError(f"{self.id_field} must be an integer or string")
        return int(value)

    def _row_path(self, row_id: int) -> Path:
        return self.rows_path / f"{row_id:08d}.json"

    def load(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        seen: set[int] = set()
        for row_path in sorted(self.rows_path.glob("*.json")):
            raw: Any = json.loads(row_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
                raise TypeError(f"Checkpoint row must be an object: {row_path}")
            row = dict(raw)
            row_id = self._id(row)
            if row_id in seen or row_path != self._row_path(row_id):
                raise ValueError(f"Invalid or duplicate checkpoint row: {row_path}")
            seen.add(row_id)
            rows.append(row)
        return rows

    def completed_ids(self) -> set[int]:
        return {self._id(row) for row in self.load()}

    def write(self, row: dict[str, object]) -> None:
        row_id = self._id(row)
        destination = self._row_path(row_id)
        if destination.exists():
            actual = json.loads(destination.read_text(encoding="utf-8"))
            if actual != row:
                raise ValueError(f"Checkpoint id={row_id} already has different content")
            return
        write_json_atomic(destination, row)

    def consolidate(self, output: Path) -> list[dict[str, object]]:
        rows = sorted(self.load(), key=self._id)
        write_jsonl_atomic(output, rows)
        return rows
