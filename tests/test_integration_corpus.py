from __future__ import annotations

from pathlib import Path

import pytest

from vifinqa.evidence.store import TableStore, parsed_table_to_long_frame
from vifinqa.programs.compiler import compile_expression
from vifinqa.programs.executor import execute_expression
from vifinqa.programs.ir import BinaryExpr, CellExpr, LiteralExpr

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/raw/ViFinQA"
MANIFEST = ROOT / "data/processed/table_manifest.parquet"
TABLE_REF = "VJC_financial_statements_2018_separate|table_50"
USD_TABLE_REF = "ACV_financial_statements_2018_separate|table_38"
UNIT_ABLATION_REF = "AAA_financial_statements_2015_consolidated|table_6"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not DATA_ROOT.exists() or not MANIFEST.exists(), reason="local ViFinQA corpus is optional"
    ),
]


def test_real_question_one_evidence_executes_in_requested_unit() -> None:
    store = TableStore.from_parquet(DATA_ROOT, MANIFEST, {TABLE_REF})
    record, table = store.load(TABLE_REF)
    frame = parsed_table_to_long_frame(record, table)
    query = compile_expression(
        BinaryExpr(
            "/",
            CellExpr("df1", row_index=0, column_index=1),
            LiteralExpr(1_000_000.0),
        )
    )
    assert execute_expression(query, {"df1": frame}) == pytest.approx(208_253.201298)


def test_mixed_currency_row_gets_cell_level_usd_unit() -> None:
    store = TableStore.from_parquet(DATA_ROOT, MANIFEST, {USD_TABLE_REF})
    record, table = store.load(USD_TABLE_REF)
    frame = parsed_table_to_long_frame(record, table)
    cell = frame.loc[(frame["row_index"] == 0) & (frame["column_index"] == 1)].iloc[0]
    # The report declares VND as the default for this run of tables; the mixed USD row still
    # overrides it at cell level.
    assert table.unit == "VND"
    assert cell["source_unit"] == "USD"
    assert float(cell["base_value"]) / 1_000_000 == pytest.approx(6.15569834)


def test_table_store_can_replay_frozen_manifest_units_for_ablation() -> None:
    store = TableStore.from_parquet(DATA_ROOT, MANIFEST, {UNIT_ABLATION_REF})
    record, latest = store.load(UNIT_ABLATION_REF)
    _, frozen = store.load(UNIT_ABLATION_REF, unit_source="manifest")

    assert record.unit == "THOUSAND_VND"
    assert latest.unit == "VND"
    assert frozen.unit == record.unit
    with pytest.raises(ValueError, match="Unknown table unit source"):
        store.load(UNIT_ABLATION_REF, unit_source="invalid")
