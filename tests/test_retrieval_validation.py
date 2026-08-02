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
