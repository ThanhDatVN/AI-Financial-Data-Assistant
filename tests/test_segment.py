from __future__ import annotations

from vifinqa.parsing.segment import segment_tables, table_markup_counts


def test_segment_retains_all_position_identifiers() -> None:
    text = """===== PAGE 1 =====
Đơn vị: triệu đồng
<table><tr><td>A</td></tr></table>
===== PAGE 2 =====
5.1. Tiền và tương đương tiền
<table>
<tr><td>B</td></tr>
</table>
"""
    tables = segment_tables(text)
    assert [table.table_id for table in tables] == [1, 2]
    assert [table.page_no for table in tables] == [1, 2]
    assert tables[0].line_no == 3
    assert tables[1].section_title == "5.1. Tiền và tương đương tiền"
    assert len(tables[1].context_before) == 1
    assert tables[1].context_before[0] == tables[1].section_title
    assert table_markup_counts(text) == (2, 2, 2)


def test_numbered_section_persists_across_a_previous_table_boundary() -> None:
    text = """===== PAGE 1 =====
5.1. Cash and cash equivalents
<table><tr><td>A</td></tr></table>
Continuation
<table><tr><td>B</td></tr></table>
"""
    first, second = segment_tables(text)
    assert first.section_title == "5.1. Cash and cash equivalents"
    assert second.context_before == ("Continuation",)
    assert second.section_title == first.section_title
