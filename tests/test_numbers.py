from __future__ import annotations

import pytest

from vifinqa.parsing.numbers import parse_financial_number


@pytest.mark.parametrize(
    ("raw", "expected", "is_dash"),
    [
        ("1.071.561.008.455", 1_071_561_008_455.0, False),
        ("(1.234.567)", -1_234_567.0, False),
        ("1.234,56", 1234.56, False),
        ("1,234.56", 1234.56, False),
        ("12,5%", 12.5, False),
        ("-", 0.0, True),
    ],
)
def test_parse_financial_number(raw: str, expected: float, is_dash: bool) -> None:
    parsed = parse_financial_number(raw)
    assert parsed.value == expected
    assert parsed.is_dash is is_dash


def test_parse_financial_number_does_not_guess_ocr_text() -> None:
    assert parse_financial_number("1O0.000").value is None
