from __future__ import annotations

from importlib import import_module

import pytest

overlay = import_module("scripts.53_overlay_generation_subset")


def _row(question_id: int, answer: float) -> dict[str, object]:
    return {"id": question_id, "answer": answer}


def test_overlay_predictions_replaces_only_fixed_subset() -> None:
    baseline = [_row(1, 1.0), _row(2, 2.0), _row(3, 3.0)]
    variant = [_row(3, 30.0), _row(1, 10.0)]

    merged = overlay.overlay_predictions(baseline, variant, {1, 3})

    assert [(row["id"], row["answer"]) for row in merged] == [
        (1, 10.0),
        (2, 2.0),
        (3, 30.0),
    ]


def test_overlay_predictions_rejects_partial_or_extra_variant() -> None:
    baseline = [_row(1, 1.0), _row(2, 2.0)]

    with pytest.raises(ValueError, match="Variant ID mismatch"):
        overlay.overlay_predictions(baseline, [_row(1, 10.0)], {1, 2})
