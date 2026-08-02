from __future__ import annotations

from pathlib import Path

from vifinqa.indexing.manifest import record_from_table
from vifinqa.parsing.models import RawTable
from vifinqa.parsing.table_parser import parse_table


def test_manifest_keeps_identity_and_excludes_numeric_values_from_retrieval_text() -> None:
    raw = RawTable(
        table_id=3,
        page_no=2,
        line_no=20,
        char_offset=100,
        html=(
            "<table><tr><td>Chỉ tiêu</td><td>2024</td></tr>"
            "<tr><td>Doanh thu thuần</td><td>1.234.567</td></tr></table>"
        ),
        context_before=("Đơn vị: triệu đồng",),
        section_title="Báo cáo kết quả kinh doanh",
    )
    record = record_from_table(
        parse_table(raw),
        doc_id="AAA_2024_consolidated",
        ticker="AAA",
        year=2024,
        scope="consolidated",
        source_path=Path("financial_statements/AAA/report.txt"),
    )
    assert record.table_ref == "AAA_2024_consolidated|table_3"
    assert "Doanh thu thuần" in record.retrieval_text
    assert "1.234.567" not in record.retrieval_text
    assert len(record.html_sha256) == 64
