from __future__ import annotations

from importlib import import_module

import pytest

sampler = import_module("scripts.64_sample_experiment_subset")


def _row(question_id: int, tickers: int, years: int) -> dict[str, object]:
    return {
        "id": question_id,
        "query_spec": {
            "tickers": [f"T{index}" for index in range(tickers)],
            "years": list(range(years)),
        },
    }


def test_proportional_allocation_uses_largest_remainders() -> None:
    counts = {"1": 441, "2": 141, "3": 119, "4": 127, "5": 76, "6": 34, "7": 14, "8+": 60}

    allocated = sampler.proportional_allocation(counts, 200)

    assert allocated == {"1": 87, "2": 28, "3": 23, "4": 25, "5": 15, "6": 7, "7": 3, "8+": 12}


def test_sample_ids_is_deterministic_and_stratified() -> None:
    rows = [_row(index, 1, 1) for index in range(1, 11)]
    rows += [_row(index, 2, 2) for index in range(11, 21)]

    first, allocation = sampler.sample_ids(rows, size=6, seed=17)
    second, _ = sampler.sample_ids(rows, size=6, seed=17)

    assert first == second
    assert len(first) == len(set(first)) == 6
    assert allocation["1"] == 3
    assert allocation["4"] == 3


def test_sample_ids_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="Duplicate retrieval ID"):
        sampler.sample_ids([_row(1, 1, 1), _row(1, 2, 1)], size=1, seed=0)
