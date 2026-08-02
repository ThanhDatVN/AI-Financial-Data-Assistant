from __future__ import annotations

from vifinqa.parsing.units import detect_source_unit, resolve_cell_unit, to_base_unit


def test_usd_units_and_scale_are_detected() -> None:
    assert detect_source_unit(("Don vi: trieu USD",)) == "MILLION_USD"
    assert detect_source_unit(("Currency: USD",)) == "USD"
    assert to_base_unit(2.5, "MILLION_USD") == 2_500_000.0


def test_cell_unit_overrides_table_default_for_mixed_unit_table() -> None:
    assert (
        resolve_cell_unit(
            "MILLION_VND",
            row_label='US Dollar ("USD")',
            column_label="Closing balance",
        )
        == "USD"
    )
    assert (
        resolve_cell_unit(
            "MILLION_VND",
            row_label="State shareholder",
            column_label="Closing balance | %",
        )
        == "PERCENT"
    )
