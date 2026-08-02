from __future__ import annotations

import pytest

from vifinqa.eval.metrics import answer_is_correct, retrieval_metrics


def test_f2_formula() -> None:
    metrics = retrieval_metrics(["gold", "extra"], {"gold"})
    assert metrics.precision == 0.5
    assert metrics.recall == 1.0
    assert metrics.f2 == pytest.approx(5 / 6)


def test_answer_tolerance_is_absolute() -> None:
    assert answer_is_correct(10.0, 10.01)
    assert not answer_is_correct(10.0, 10.011)
