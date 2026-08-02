from __future__ import annotations

from vifinqa.retrieval.fusion import coverage_budget


def test_candidate_budget_never_truncates_metadata_routes() -> None:
    assert coverage_budget(20, 7) == 20
    assert coverage_budget(20, 30) == 30
