from __future__ import annotations

import importlib

import pytest

evaluator = importlib.import_module("scripts.62_evaluate_retrieval")


def test_retrieval_evaluator_reports_complete_evidence() -> None:
    qrels = [
        {
            "id": 1,
            "status": "reviewed",
            "answerability": "ANSWERABLE",
            "gold_tables": ["a", "b"],
            "stratum": "multi",
        },
        {
            "id": 2,
            "status": "adjudicated",
            "answerability": "ANSWERABLE",
            "gold_tables": ["c"],
            "stratum": "single",
        },
    ]
    run = [
        {"id": 1, "fused": ["a", "x", "b"]},
        {"id": 2, "fused": ["x", "c"]},
    ]
    report = evaluator.evaluate(
        qrels,
        run,
        rankings=["fused"],
        cutoffs=[1, 3],
        statuses={"reviewed", "adjudicated"},
    )
    fused = report["rankings"]["fused"]
    assert fused["1"]["hit_rate"] == 0.5
    assert fused["3"]["complete_evidence_rate"] == 1.0
    assert report["slice_counts"] == {"multi": 1, "single": 1}


def test_retrieval_evaluator_refuses_unlabeled_templates() -> None:
    with pytest.raises(ValueError, match="No reviewed"):
        evaluator.evaluate(
            [{"id": 1, "status": "unlabeled", "gold_tables": []}],
            [{"id": 1, "fused": []}],
            rankings=["fused"],
            cutoffs=[10],
            statuses={"reviewed"},
        )
