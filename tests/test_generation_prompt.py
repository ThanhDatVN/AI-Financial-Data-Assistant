from __future__ import annotations

from pathlib import Path

from vifinqa.generation.prompt import CandidateSchema, build_program_prompt
from vifinqa.indexing.manifest import ManifestRecord
from vifinqa.parsing.models import RawTable
from vifinqa.parsing.table_parser import parse_table


def test_program_prompt_contains_coordinates_but_hides_source_numbers() -> None:
    raw = RawTable(
        1,
        1,
        1,
        0,
        (
            "<table><tr><td>Chỉ tiêu</td><td>2024</td></tr>"
            "<tr><td>Doanh thu</td><td>123456789</td></tr></table>"
        ),
        ("Đơn vị: VND",),
        None,
    )
    table = parse_table(raw)
    record = ManifestRecord(
        "DOC|table_1",
        "DOC",
        "AAA",
        2024,
        "consolidated",
        1,
        1,
        1,
        0,
        None,
        "VND",
        1,
        2,
        2,
        table.headers,
        ("Doanh thu",),
        "",
        Path("report.txt").as_posix(),
        "0" * 64,
    )
    system, user = build_program_prompt(
        "Doanh thu năm 2024?",
        [CandidateSchema("df1", record, table)],
        target_unit="MILLION_VND",
        target_divisor=1e6,
    )
    assert "r0: Doanh thu" in user
    assert "c1: 2024" in user
    assert "123456789" not in user
    assert "typed arithmetic IR" in system
    assert "never emit Python/Pandas code" in system
