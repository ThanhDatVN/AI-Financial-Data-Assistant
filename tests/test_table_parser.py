from __future__ import annotations

from vifinqa.parsing.models import RawTable
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
