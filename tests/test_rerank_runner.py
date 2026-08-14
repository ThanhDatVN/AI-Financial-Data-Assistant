from __future__ import annotations

import pytest

# The dense and rerank stacks need torch and sentence-transformers, which the local
# environment deliberately does not install -- they are exercised on the Linux/GPU
# runtime instead. Skipping keeps `pytest tests` runnable here; erroring on collection
# made a clean checkout look broken.
pytest.importorskip("sentence_transformers")

import importlib
import json
import sys
from pathlib import Path
from typing import Any

from vifinqa.indexing.manifest import ManifestRecord
from vifinqa.retrieval.rerank import RerankedHit

runner = importlib.import_module("scripts.33_rerank_retrieval")


def _record(ref: str) -> ManifestRecord:
    return ManifestRecord(
        table_ref=ref,
        doc_id=ref.split("|", 1)[0],
        ticker="AAA",
        report_year=2024,
        scope="consolidated",
        table_id=1,
        page_no=1,
        line_no=1,
        char_offset=1,
        section_title="Section",
        unit="VND",
        header_rows=1,
        n_rows=1,
        n_cols=2,
        headers=("Metric", "2024"),
        row_labels=("Revenue",),
        retrieval_text=ref,
        source_path="report.txt",
        html_sha256="0" * 64,
    )


class FakeReranker:
    instances = 0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        FakeReranker.instances += 1

    def rerank(
        self,
        question: str,
        candidates: list[ManifestRecord],
        *,
        top_k: int,
        batch_size: int,
        max_batch_tokens: int,
    ) -> list[RerankedHit]:
        ordered = list(reversed(candidates))[:top_k]
        return [
            RerankedHit(record.table_ref, float(index), index)
            for index, record in enumerate(ordered, 1)
        ]


def test_rerank_runner_writes_resume_safe_rows(tmp_path: Path, monkeypatch: Any) -> None:
    retrieval = tmp_path / "hybrid.jsonl"
    retrieval.write_text(
        json.dumps(
            {
                "id": 1,
                "question": "Doanh thu AAA 2024?",
                "query_spec": {
                    "tickers": ["AAA"],
                    "years": [2024],
                    "scope": "consolidated",
                    "target_unit": "VND",
                    "target_divisor": 1.0,
                },
                "bm25": ["AAA|table_1"],
                "dense": ["AAA|table_2"],
                "hybrid": ["AAA|table_1", "AAA|table_2"],
                "fused": ["AAA|table_1", "AAA|table_2"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    records = tmp_path / "records.jsonl"
    records.write_text(
        "\n".join(_record(ref).to_json() for ref in ("AAA|table_1", "AAA|table_2")) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "shard_0"
    monkeypatch.setattr(runner, "CrossEncoderReranker", FakeReranker)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "33_rerank_retrieval.py",
            "--retrieval",
            str(retrieval),
            "--records",
            str(records),
            "--output",
            str(output),
            "--candidate-tables",
            "2",
            "--top-k",
            "1",
        ],
    )
    runner.main()
    first_instances = FakeReranker.instances
    runner.main()

    row = json.loads((output / "retrieval.jsonl").read_text(encoding="utf-8"))
    assert row["reranked"] == ["AAA|table_2"]
    assert row["fused"] == row["reranked"]
    assert len(row["reranker_scores"]) == 2
    assert FakeReranker.instances == first_instances
