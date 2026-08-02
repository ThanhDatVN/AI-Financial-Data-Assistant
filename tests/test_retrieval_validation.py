from __future__ import annotations

from importlib import import_module

validator = import_module("scripts.32_validate_retrieval")


def test_route_matching_respects_scope_and_unknown_fallback() -> None:
    metadata = ("AAA", 2024, "unknown")
    assert validator._route_matches(metadata, ticker="AAA", year=2024, scope="separate")
    assert not validator._route_matches(metadata, ticker="BBB", year=2024, scope="separate")
    assert not validator._route_matches(
        ("AAA", 2023, "separate"), ticker="AAA", year=2024, scope=None
    )


def test_reranker_integrity_requires_scores_for_the_hybrid_pool() -> None:
    row: dict[str, object] = {
        "hybrid": ["a", "b", "c"],
        "reranked": ["b", "a"],
        "fused": ["b", "a"],
        "reranker_scores": [
            {"table_ref": "b", "score": 2.0},
            {"table_ref": "a", "score": 1.0},
            {"table_ref": "c", "score": 0.0},
        ],
    }
    assert validator._reranker_error(row) is None
    row["reranker_scores"] = [{"table_ref": "b", "score": 2.0}]
    assert "every hybrid ref" in validator._reranker_error(row)
