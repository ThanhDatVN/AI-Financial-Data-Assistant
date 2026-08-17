from __future__ import annotations

from pathlib import Path

from vifinqa.indexing.manifest import ManifestRecord
from vifinqa.retrieval.routing import NO_ROUTING, scope_routed


def _record(ticker: str, year: int, scope: str, table: int) -> ManifestRecord:
    doc = f"{ticker}_financial_statements_{year}_{scope}"
    return ManifestRecord(
        table_ref=f"{doc}|table_{table}",
        doc_id=doc,
        ticker=ticker,
        report_year=year,
        scope=scope,
        table_id=table,
        page_no=1,
        line_no=table * 10,
        char_offset=0,
        section_title=None,
        unit="VND",
        header_rows=1,
        n_rows=2,
        n_cols=2,
        headers=("Chỉ tiêu", str(year)),
        row_labels=("Doanh thu thuần",),
        retrieval_text="Doanh thu thuần",
        source_path=Path("report.txt").as_posix(),
        html_sha256="0" * 64,
    )


def _records(*records: ManifestRecord) -> dict[str, ManifestRecord]:
    return {record.table_ref: record for record in records}


def test_unstated_scope_drops_the_other_statement_and_keeps_the_order() -> None:
    separate = _record("AAA", 2024, "separate", 1)
    consolidated = _record("AAA", 2024, "consolidated", 2)
    deeper = _record("AAA", 2024, "consolidated", 3)
    ranking = [separate.table_ref, consolidated.table_ref, deeper.table_ref]
    kept = scope_routed(
        ranking,
        records=_records(separate, consolidated, deeper),
        scope=None,
        policy="consolidated",
    )
    assert kept == [consolidated.table_ref, deeper.table_ref]


def test_stated_scope_wins_over_the_policy() -> None:
    separate = _record("AAA", 2024, "separate", 1)
    consolidated = _record("AAA", 2024, "consolidated", 2)
    kept = scope_routed(
        [separate.table_ref, consolidated.table_ref],
        records=_records(separate, consolidated),
        scope="separate",
        policy="consolidated",
    )
    assert kept == [separate.table_ref]


def test_a_ticker_year_with_only_the_other_statement_keeps_it() -> None:
    """Some companies only ever filed one of the two, and a filter must not empty the prompt."""
    only_separate = _record("BBB", 2019, "separate", 1)
    elsewhere = _record("AAA", 2024, "consolidated", 1)
    kept = scope_routed(
        [only_separate.table_ref, elsewhere.table_ref],
        records=_records(only_separate, elsewhere),
        scope=None,
        policy="consolidated",
    )
    assert kept == [only_separate.table_ref, elsewhere.table_ref]


def test_the_guard_is_per_ticker_year_not_per_question() -> None:
    """A cohort question must not keep one company's separate report because another has both."""
    both_a = _record("AAA", 2024, "separate", 1)
    both_b = _record("AAA", 2024, "consolidated", 2)
    only_separate = _record("BBB", 2024, "separate", 1)
    kept = scope_routed(
        [both_a.table_ref, only_separate.table_ref, both_b.table_ref],
        records=_records(both_a, both_b, only_separate),
        scope=None,
        policy="consolidated",
    )
    assert kept == [only_separate.table_ref, both_b.table_ref]


def test_an_unreadable_scope_is_not_evidence_of_the_wrong_one() -> None:
    unknown = _record("CCC", 2023, "unknown", 1)
    consolidated = _record("CCC", 2023, "consolidated", 2)
    kept = scope_routed(
        [unknown.table_ref, consolidated.table_ref],
        records=_records(unknown, consolidated),
        scope=None,
        policy="consolidated",
    )
    assert kept == [unknown.table_ref, consolidated.table_ref]


def test_a_table_missing_from_the_manifest_survives_the_filter() -> None:
    consolidated = _record("AAA", 2024, "consolidated", 1)
    kept = scope_routed(
        ["mystery|table_9", consolidated.table_ref],
        records=_records(consolidated),
        scope=None,
        policy="consolidated",
    )
    assert kept == ["mystery|table_9", consolidated.table_ref]


def test_the_default_policy_changes_nothing() -> None:
    separate = _record("AAA", 2024, "separate", 1)
    consolidated = _record("AAA", 2024, "consolidated", 2)
    ranking = [separate.table_ref, consolidated.table_ref]
    assert scope_routed(ranking, records=_records(separate, consolidated), scope=None) == ranking
    assert (
        scope_routed(
            ranking, records=_records(separate, consolidated), scope=None, policy=NO_ROUTING
        )
        == ranking
    )
