from __future__ import annotations

from vifinqa.parsing.models import RawTable
from vifinqa.parsing.segment import segment_tables
from vifinqa.parsing.table_parser import html_to_matrix, parse_table


def test_html_to_matrix_expands_rowspan_and_colspan() -> None:
    html = """
    <table>
      <tr><th rowspan="2">Chỉ tiêu</th><th colspan="2">Kỳ</th></tr>
      <tr><th>2024</th><th>2023</th></tr>
      <tr><td>Doanh thu</td><td>10</td><td>9</td></tr>
    </table>
    """
    assert html_to_matrix(html) == (
        ("Chỉ tiêu", "Kỳ", "Kỳ"),
        ("Chỉ tiêu", "2024", "2023"),
        ("Doanh thu", "10", "9"),
    )


def test_parse_table_combines_headers_and_detects_unit() -> None:
    raw = RawTable(
        table_id=1,
        page_no=1,
        line_no=2,
        char_offset=10,
        html=(
            "<table><tr><td>Chỉ tiêu</td><td>2024</td></tr>"
            "<tr><td>Doanh thu</td><td>1.000</td></tr></table>"
        ),
        context_before=("Đơn vị: triệu đồng",),
        section_title=None,
    )
    parsed = parse_table(raw)
    assert parsed.headers == ("Chỉ tiêu", "2024")
    assert parsed.rows == (("Doanh thu", "1.000"),)
    assert parsed.unit == "MILLION_VND"


def test_unit_declaration_persists_across_intervening_tables() -> None:
    text = (
        "Đơn vị tính: VND\n"
        "<table><tr><td>A</td><td>1</td></tr></table>\n"
        "7. Các khoản đầu tư\n"
        "<table><tr><td>Giá gốc</td><td>10</td></tr></table>"
    )
    raw_tables = segment_tables(text)

    assert len(raw_tables) == 2
    assert raw_tables[1].context_before == ("7. Các khoản đầu tư",)
    assert raw_tables[1].unit_declaration == "Đơn vị tính: VND"
    assert parse_table(raw_tables[1]).unit == "VND"
