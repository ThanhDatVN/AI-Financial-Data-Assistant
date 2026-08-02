from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from vifinqa.indexing.manifest import ManifestRecord
from vifinqa.retrieval.bm25 import BM25Index
from vifinqa.retrieval.fusion import balanced_round_robin, reciprocal_rank_fusion


def _record(ref: str, text: str, ticker: str = "AAA") -> ManifestRecord:
    return ManifestRecord(
        table_ref=ref,
        doc_id=ref.split("|")[0],
        ticker=ticker,
        report_year=2024,
        scope="consolidated",
        table_id=1,
        page_no=1,
        line_no=1,
        char_offset=0,
        section_title=None,
        unit="VND",
        header_rows=1,
        n_rows=2,
        n_cols=2,
        headers=("Chỉ tiêu", "2024"),
        row_labels=(text,),
        retrieval_text=text,
        source_path=Path("report.txt").as_posix(),
        html_sha256="0" * 64,
    )


def test_bm25_is_accent_tolerant_and_respects_metadata() -> None:
    revenue = _record("A|table_1", "Doanh thu thuần")
    assets = replace(_record("B|table_1", "Tổng tài sản"), ticker="BBB")
    distractor = _record("C|table_1", "Tài sản tài sản tài sản")
    index = BM25Index.build([revenue, assets, distractor])
    assert index.search("doanh thu thuan", top_k=1)[0].table_ref == revenue.table_ref
    assert (
        index.search("tài sản", top_k=1, candidate_k=1, tickers={"BBB"})[0].table_ref
        == assets.table_ref
    )


def test_bm25_returns_empty_for_impossible_metadata_route() -> None:
    index = BM25Index.build([_record("A|table_1", "Doanh thu thuần")])
    assert index.search("doanh thu", tickers={"MISSING"}) == []


def test_rrf_is_deterministic_and_deduplicates_each_ranking() -> None:
    fused = reciprocal_rank_fusion([["A", "A", "B"], ["B", "A"]], rrf_k=0)
    assert [item for item, _ in fused] == ["A", "B"]


def test_balanced_round_robin_preserves_each_entity_route() -> None:
    assert balanced_round_robin([["A1", "A2"], ["B1", "B2"]], limit=3) == [
        "A1",
        "B1",
        "A2",
    ]
