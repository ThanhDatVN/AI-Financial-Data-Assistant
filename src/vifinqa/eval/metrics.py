from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    precision: float
    recall: float
    f2: float


def retrieval_metrics(
    retrieved: list[str], gold: set[str], *, beta: float = 2.0
) -> RetrievalMetrics:
    retrieved_set = set(retrieved)
    true_positive = len(retrieved_set & gold)
    precision = true_positive / len(retrieved_set) if retrieved_set else 0.0
    recall = true_positive / len(gold) if gold else 0.0
    beta2 = beta * beta
    denominator = beta2 * precision + recall
    f_beta = (1 + beta2) * precision * recall / denominator if denominator else 0.0
    return RetrievalMetrics(precision, recall, f_beta)


def answer_is_correct(expected: object, actual: object, *, abs_tolerance: float = 0.01) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return False
    try:
        expected_float = float(expected)  # type: ignore[arg-type]
        actual_float = float(actual)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return math.isclose(expected_float, actual_float, rel_tol=0.0, abs_tol=abs_tolerance)
