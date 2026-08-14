from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest

finalize = import_module("scripts.45_finalize_submission")


def test_finalizer_applies_semantics_then_z4_then_line_refs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw.json"
    retrieval = tmp_path / "retrieval.jsonl"
    manifest = tmp_path / "manifest.parquet"
    output = tmp_path / "final.json"
    for path in (raw, retrieval, manifest):
        path.write_text("[]\n", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(command: list[object], *, check: bool) -> SimpleNamespace:
        assert check
        normalized = [str(item) for item in command]
        calls.append(normalized)
        target = Path(normalized[normalized.index("--output") + 1])
        target.write_text("[]\n", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(finalize.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "45_finalize_submission.py",
            str(raw),
            "--retrieval",
            str(retrieval),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
        ],
    )

    finalize.main()

    assert [Path(call[1]).name for call in calls] == [
        "44_apply_semantic_conventions.py",
        "43_widen_citations.py",
        "42_retarget_table_refs.py",
    ]
    assert "--tables" in calls[1] and "5" in calls[1]
    assert "--table-router" in calls[1] and "consolidated" in calls[1]
    assert "--router-only" in calls[1]
    assert "--grammar" in calls[2] and "line" in calls[2]
    assert output.read_text(encoding="utf-8") == "[]\n"
