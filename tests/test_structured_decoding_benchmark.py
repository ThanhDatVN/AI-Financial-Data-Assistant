from __future__ import annotations

from importlib import import_module
from typing import Any

import pytest

benchmark = import_module("scripts.72_benchmark_structured_decoding")


def _measurement(mode: str, rate: float, *, valid_json: bool = True) -> Any:
    return benchmark.Measurement(
        question_id=1,
        repeat=0,
        mode=mode,
        wall_seconds=2.0,
        ttft_seconds=0.5,
        decode_seconds=1.5,
        completion_tokens=10,
        decode_tokens_per_second=rate,
        content_characters=20,
        valid_json=valid_json,
        error=None,
    )


def test_comparison_flags_a_large_guided_decoding_gap() -> None:
    rows = [
        _measurement("json_schema", 10.0),
        _measurement("unconstrained", 20.0, valid_json=False),
    ]
    summaries = [
        benchmark._summary(rows, "json_schema"),
        benchmark._summary(rows, "unconstrained"),
    ]

    result = benchmark._comparison(summaries)

    assert result["unconstrained_over_json_schema_decode_speed"] == 2.0
    assert result["unconstrained_valid_json_rate"] == 0.0
    assert result["diagnosis"] == "guided_decoding_is_a_likely_bottleneck"


def test_comparison_does_not_blame_guided_decoding_for_a_small_gap() -> None:
    rows = [
        _measurement("json_schema", 10.0),
        _measurement("unconstrained", 11.0),
    ]
    summaries = [
        benchmark._summary(rows, "json_schema"),
        benchmark._summary(rows, "unconstrained"),
    ]

    assert benchmark._comparison(summaries)["diagnosis"] == (
        "guided_decoding_is_not_the_primary_bottleneck"
    )


def test_prompt_builder_uses_the_selected_table_unit_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeStore:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def load(self, table_ref: str, *, unit_source: str) -> tuple[object, object]:
            self.calls.append((table_ref, unit_source))
            return object(), object()

    store = FakeStore()
    monkeypatch.setattr(benchmark, "parsed_table_to_long_frame", lambda _record, _table: object())
    monkeypatch.setattr(benchmark, "numeric_cells_of", lambda _frame: [object()])
    monkeypatch.setattr(benchmark, "build_program_prompt", lambda *_args, **_kwargs: ("s", "u"))

    prompts = benchmark._build_prompts(
        [
            {
                "id": 1,
                "question": "q",
                "fused": ["doc::table::1"],
                "query_spec": {
                    "tickers": [],
                    "years": [],
                    "target_unit": "VND",
                    "target_divisor": 1.0,
                },
            }
        ],
        store=store,
        candidate_tables=1,
        table_unit_source="manifest",
    )

    assert [prompt.question_id for prompt in prompts] == [1]
    assert store.calls == [("doc::table::1", "manifest")]
