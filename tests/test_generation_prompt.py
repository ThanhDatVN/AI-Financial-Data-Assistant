from __future__ import annotations

from pathlib import Path

from vifinqa.evidence.store import parsed_table_to_long_frame
from vifinqa.generation.prompt import CandidateSchema, build_program_prompt, numeric_cells_of
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
    frame = parsed_table_to_long_frame(record, table)
    system, user = build_program_prompt(
        "Doanh thu năm 2024?",
        [CandidateSchema("df1", record, table, numeric_cells_of(frame))],
        target_unit="MILLION_VND",
        target_divisor=1e6,
    )
    assert "r0: Doanh thu" in user
    assert "c1: 2024" in user
    assert "123456789" not in user
    assert "typed arithmetic IR" in system
    assert "never emit Python/Pandas code" in system

    # Where the numbers sit is structure the model needs to pick a coordinate at all; the
    # numbers themselves stay hidden.
    assert "r0: Doanh thu  -> values at c1" in user
    assert numeric_cells_of(frame) == frozenset({(0, 1)})


def test_program_prompt_states_the_cohort_the_program_has_to_cover() -> None:
    raw = RawTable(1, 1, 1, 0, "<table><tr><td>Doanh thu</td><td>1</td></tr></table>", (), None)
    table = parse_table(raw)
    record = ManifestRecord(
        "DOC|table_1",
        "DOC",
        "VIC",
        2025,
        "consolidated",
        1,
        1,
        1,
        0,
        None,
        "VND",
        1,
        1,
        2,
        table.headers,
        ("Doanh thu",),
        "",
        Path("report.txt").as_posix(),
        "0" * 64,
    )
    _, user = build_program_prompt(
        "Trong nhóm CEO, DIG và VIC, doanh nghiệp nào có doanh thu cao nhất?",
        [CandidateSchema("df1", record, table)],
        target_unit="MILLION_VND",
        target_divisor=1e6,
        required_tickers=["CEO", "DIG", "VIC"],
        required_years=[2024, 2025],
    )
    # The resolver already knows the group; a program covering one member is rejected, so
    # the requirement belongs in the prompt rather than in the Vietnamese sentence alone.
    assert "CEO, DIG, VIC" in user
    assert "2024, 2025" in user
    assert "at least one cell from every one of these tickers" in user


def test_program_prompt_shows_the_section_that_separates_restatements() -> None:
    raw = RawTable(
        1,
        1,
        1,
        0,
        "<table><tr><td>Lãi tiền gửi</td><td>1</td></tr></table>",
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
        "29. Doanh thu hoạt động tài chính",
        "VND",
        1,
        1,
        2,
        table.headers,
        ("Lãi tiền gửi",),
        "",
        Path("report.txt").as_posix(),
        "0" * 64,
    )
    _, user = build_program_prompt(
        "Lãi tiền gửi?",
        [CandidateSchema("df1", record, table)],
        target_unit="MILLION_VND",
        target_divisor=1e6,
    )
    assert 'section="29. Doanh thu hoạt động tài chính"' in user
    # Without the map the rows render exactly as before, so old callers are unaffected.
    rendered_rows = user.split("rows:\n", 1)[1].split("</table>", 1)[0]
    assert "-> values at" not in rendered_rows
