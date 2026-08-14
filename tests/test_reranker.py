from __future__ import annotations

import pytest

# The dense and rerank stacks need torch and sentence-transformers, which the local
# environment deliberately does not install -- they are exercised on the Linux/GPU
# runtime instead. Skipping keeps `pytest tests` runnable here; erroring on collection
# made a clean checkout look broken.
pytest.importorskip("sentence_transformers")

from typing import Any

from vifinqa.indexing.manifest import ManifestRecord
from vifinqa.retrieval import rerank as rerank_module
from vifinqa.retrieval.rerank import CrossEncoderReranker, RerankedHit, preserve_route_coverage


def _record(
    ref: str,
    ticker: str,
    year: int,
    text: str = "original\nkhong dau: duplicate",
) -> ManifestRecord:
    return ManifestRecord(
        table_ref=ref,
        doc_id=ref.split("|", 1)[0],
        ticker=ticker,
        report_year=year,
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
        headers=("Metric", str(year)),
        row_labels=("Revenue",),
        retrieval_text=text,
        source_path="report.txt",
        html_sha256="0" * 64,
    )


class FakeModel:
    def __init__(self) -> None:
        self.half_called = False

    def half(self) -> None:
        self.half_called = True


class FakeCrossEncoder:
    init_kwargs: dict[str, Any] = {}
    pairs: list[tuple[str, str]] = []
    predict_batch_sizes: list[int] = []

    def __init__(self, model_id: str, **kwargs: Any) -> None:
        self.init_kwargs = {"model_id": model_id, **kwargs}
        FakeCrossEncoder.init_kwargs = self.init_kwargs
        self.model = FakeModel()

    def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> list[float]:
        FakeCrossEncoder.pairs = pairs
        FakeCrossEncoder.predict_batch_sizes.append(int(kwargs["batch_size"]))
        return [0.9 if "second" in pair[1] else 0.1 for pair in pairs]

    def tokenizer(
        self,
        questions: list[str],
        documents: list[str],
        **kwargs: Any,
    ) -> dict[str, list[int]]:
        return {"length": [10 if "short" in document else 100 for document in documents]}


def test_cross_encoder_is_pinned_and_does_not_score_duplicate_ascii_view(monkeypatch: Any) -> None:
    monkeypatch.setattr(rerank_module, "CrossEncoder", FakeCrossEncoder)
    reranker = CrossEncoderReranker(
        "model",
        model_revision="a" * 40,
        device="cuda:1",
        max_length=4096,
        use_fp16=True,
    )
    FakeCrossEncoder.predict_batch_sizes.clear()
    hits = reranker.rerank(
        "question",
        [
            _record("a|table_1", "A", 2023, "first\nkhong dau: duplicate"),
            _record("b|table_1", "B", 2024, "second\nkhong dau: duplicate"),
        ],
    )

    assert FakeCrossEncoder.init_kwargs["revision"] == "a" * 40
    assert FakeCrossEncoder.init_kwargs["max_length"] == 4096
    assert reranker.model.model.half_called is True
    assert [pair[1] for pair in FakeCrossEncoder.pairs] == ["first", "second"]
    assert [hit.table_ref for hit in hits] == ["b|table_1", "a|table_1"]


def test_cross_encoder_adapts_batch_size_without_shortening_context(monkeypatch: Any) -> None:
    monkeypatch.setattr(rerank_module, "CrossEncoder", FakeCrossEncoder)
    reranker = CrossEncoderReranker("model", max_length=8192)
    FakeCrossEncoder.predict_batch_sizes.clear()
    hits = reranker.rerank(
        "question",
        [
            _record("short", "A", 2023, "short"),
            _record("long", "A", 2023, "long"),
        ],
        batch_size=4,
        max_batch_tokens=100,
    )
    assert {hit.table_ref for hit in hits} == {"short", "long"}
    assert FakeCrossEncoder.predict_batch_sizes == [1, 4]


def test_route_coverage_retains_lower_scored_required_entity_year() -> None:
    records = {
        "a23": _record("a23", "A", 2023),
        "a24": _record("a24", "A", 2024),
        "b23": _record("b23", "B", 2023),
        "b24": _record("b24", "B", 2024),
        "extra": _record("extra", "A", 2024),
    }
    hits = [
        RerankedHit("extra", 1.0, 1),
        RerankedHit("a23", 0.9, 2),
        RerankedHit("a24", 0.8, 3),
        RerankedHit("b23", 0.2, 4),
        RerankedHit("b24", 0.1, 5),
    ]
    covered = preserve_route_coverage(
        hits,
        records,
        routes=[("A", 2023), ("A", 2024), ("B", 2023), ("B", 2024)],
        scopes={"consolidated"},
        top_k=4,
    )
    assert {hit.table_ref for hit in covered} == {"a23", "extra", "b23", "b24"}
    assert [hit.rank for hit in covered] == [1, 2, 3, 4]
