from __future__ import annotations

from vifinqa.parsing.units import (
    detect_source_unit,
    detect_table_unit,
    is_unit_declaration,
    resolve_cell_unit,
    to_base_unit,
)


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


def test_table_unit_requires_a_declaration_or_consistent_headers() -> None:
    assert is_unit_declaration("Đơn vị tính: VND")
    assert not is_unit_declaration("Phát hành cổ phiếu trị giá 10 đồng")
    assert (
        detect_table_unit(
            ("Phát hành cổ phiếu trị giá 10 đồng",),
            ("Giá gốc", "Dự phòng"),
        )
        == "UNKNOWN"
    )
    assert detect_table_unit((), ("Năm 2024 VND", "Năm 2023 VND")) == "VND"
    assert detect_table_unit((), ("Tỷ lệ %", "Số lượng cổ phiếu")) == "UNKNOWN"
    assert resolve_cell_unit("VND", row_label="Công ty A", column_label="Tỷ lệ vốn") == "PERCENT"
    assert (
        resolve_cell_unit(
            "MILLION_VND",
            row_label="State shareholder",
            column_label="Closing balance | %",
        )
        == "PERCENT"
    )


def test_share_quantity_is_not_confused_with_share_value() -> None:
    assert detect_source_unit(("Số cổ phần",)) == "SHARES"
    assert detect_source_unit(("Số lượng cổ phiếu đang lưu hành",)) == "SHARES"
    assert detect_source_unit(("Giá trị cổ phần",)) == "UNKNOWN"
    assert resolve_cell_unit("VND", row_label="Cổ đông A", column_label="Giá trị cổ phần") == "VND"
    assert (
        resolve_cell_unit("VND", row_label="Cổ đông A", column_label="Mệnh giá cổ phiếu") == "VND"
    )
    assert detect_source_unit(("Lãi cơ bản trên mỗi cổ phiếu (đồng/cổ phiếu)",)) == "VND"
    assert detect_source_unit(("Lãi cơ bản (nghìn đồng/cổ phiếu)",)) == "THOUSAND_VND"
    assert (
        resolve_cell_unit(
            "MILLION_VND",
            row_label="Lợi nhuận phân bổ cho cổ đông sở hữu cổ phiếu phổ thông",
            column_label="Năm 2024",
        )
        == "MILLION_VND"
    )
