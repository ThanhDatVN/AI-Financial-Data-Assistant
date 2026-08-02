from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vifinqa.nlu.company import CompanyResolver  # noqa: E402
from vifinqa.nlu.query_spec import parse_query_spec  # noqa: E402
from vifinqa.parsing.document import statement_paths  # noqa: E402
from vifinqa.parsing.metadata import parse_document_metadata  # noqa: E402
from vifinqa.parsing.segment import segment_tables, table_markup_counts  # noqa: E402
from vifinqa.parsing.table_parser import parse_table  # noqa: E402


def percentile(values: list[int], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def load_questions(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _question_id(row: dict[str, object]) -> int:
    value = row["id"]
    if not isinstance(value, int | str):
        raise TypeError("question id must be an integer or string")
    return int(value)


def audit_questions(data_root: Path) -> dict[str, object]:
    questions = load_questions(data_root / "questions" / "questions.jsonl")
    resolver = CompanyResolver.from_csv(data_root / "code_stock.csv")
    unit_counts: Counter[str] = Counter()
    entity_count_distribution: Counter[int] = Counter()
    scope_counts: Counter[str] = Counter()
    unresolved: list[dict[str, object]] = []
    multi_year = 0
    ids: list[int] = []
    for row in questions:
        question_id = _question_id(row)
        question = str(row["question"])
        ids.append(question_id)
        spec = parse_query_spec(question, resolver)
        unit_counts[spec.target_unit] += 1
        entity_count_distribution[len(spec.entities)] += 1
        if len(spec.periods) > 1:
            multi_year += 1
        for entity in spec.entities:
            scope_counts[entity.scope or "unspecified"] += 1
        if not spec.entities:
            unresolved.append({"id": question_id, "question": question})
    return {
        "count": len(questions),
        "ids_unique": len(ids) == len(set(ids)),
        "ids_sequential": ids == list(range(1, len(ids) + 1)),
        "target_units": dict(sorted(unit_counts.items())),
        "entity_count_distribution": {
            str(key): value for key, value in sorted(entity_count_distribution.items())
        },
        "entity_scope_mentions": dict(sorted(scope_counts.items())),
        "multi_year_questions": multi_year,
        "unresolved_company_count": len(unresolved),
        "unresolved_company_examples": unresolved[:100],
    }


def audit_corpus(data_root: Path) -> dict[str, object]:
    paths = statement_paths(data_root)
    scope_counts: Counter[str] = Counter()
    year_counts: Counter[int] = Counter()
    ticker_counts: Counter[str] = Counter()
    unit_counts: Counter[str] = Counter()
    table_counts: list[int] = []
    page_counts: list[int] = []
    row_counts: list[int] = []
    col_counts: list[int] = []
    total_bytes = 0
    markup_mismatch: list[str] = []
    documents_without_tables: list[str] = []
    empty_tables = 0
    malformed_tables = 0
    sampled_tables = 0
    top_documents: list[tuple[int, str]] = []
    fiscal_year_end_detected = 0

    for index, path in enumerate(paths, start=1):
        text = path.read_text(encoding="utf-8", errors="replace")
        metadata = parse_document_metadata(path, data_root=data_root, text=text)
        raw_tables = segment_tables(text)
        openings, closings, matches = table_markup_counts(text)
        total_bytes += path.stat().st_size
        scope_counts[metadata.scope] += 1
        year_counts[metadata.year] += 1
        ticker_counts[metadata.ticker] += 1
        fiscal_year_end_detected += metadata.fiscal_year_end is not None
        page_numbers = [table.page_no for table in raw_tables if table.page_no is not None]
        page_counts.append(max(page_numbers, default=0))
        table_counts.append(len(raw_tables))
        top_documents.append((len(raw_tables), metadata.doc_id))
        if not raw_tables:
            documents_without_tables.append(metadata.doc_id)
        if not (openings == closings == matches):
            markup_mismatch.append(metadata.doc_id)
        # Full markup/position counts cover every table. Shape/unit parsing is a
        # deterministic three-table sample per document to keep the audit fast;
        # the index-build stage later parses every table and applies the same QC.
        sample_indices = (
            sorted({0, len(raw_tables) // 2, len(raw_tables) - 1})
            if index == 1 or index % 10 == 0
            else []
        )
        for sample_index in sample_indices:
            if sample_index < 0 or sample_index >= len(raw_tables):
                continue
            table = parse_table(raw_tables[sample_index])
            sampled_tables += 1
            unit_counts[table.unit] += 1
            row_counts.append(table.n_rows)
            col_counts.append(table.n_cols)
            if table.n_rows == 0 or table.n_cols == 0:
                empty_tables += 1
            if any(len(row) != table.n_cols for row in table.matrix):
                malformed_tables += 1
        if index % 100 == 0 or index == len(paths):
            print(f"audited {index}/{len(paths)} documents", flush=True)

    top_documents.sort(reverse=True)
    return {
        "documents": len(paths),
        "tickers": len(ticker_counts),
        "years": {str(key): value for key, value in sorted(year_counts.items())},
        "scopes": dict(sorted(scope_counts.items())),
        "total_bytes": total_bytes,
        "total_mib": round(total_bytes / 1024**2, 3),
        "tables": sum(table_counts),
        "tables_per_document": {
            "min": min(table_counts, default=0),
            "mean": round(statistics.fmean(table_counts), 3) if table_counts else 0.0,
            "median": statistics.median(table_counts) if table_counts else 0.0,
            "p95": round(percentile(table_counts, 0.95), 3),
            "max": max(table_counts, default=0),
        },
        "pages_per_document": {
            "median": statistics.median(page_counts) if page_counts else 0.0,
            "p95": round(percentile(page_counts, 0.95), 3),
            "max": max(page_counts, default=0),
        },
        "table_shape": {
            "rows_median": statistics.median(row_counts) if row_counts else 0.0,
            "rows_p95": round(percentile(row_counts, 0.95), 3),
            "rows_max": max(row_counts, default=0),
            "cols_median": statistics.median(col_counts) if col_counts else 0.0,
            "cols_p95": round(percentile(col_counts, 0.95), 3),
            "cols_max": max(col_counts, default=0),
        },
        "source_units": dict(sorted(unit_counts.items())),
        "source_units_sampled_tables": sampled_tables,
        "fiscal_year_end_detected": fiscal_year_end_detected,
        "empty_tables": empty_tables,
        "malformed_tables": malformed_tables,
        "documents_without_tables": documents_without_tables,
        "markup_mismatch_count": len(markup_mismatch),
        "markup_mismatch_examples": markup_mismatch[:100],
        "top_documents_by_table_count": [
            {"doc_id": doc_id, "tables": count} for count, doc_id in top_documents[:20]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/ViFinQA")
    parser.add_argument("--output", type=Path, default=ROOT / "data/interim/dataset_audit.json")
    args = parser.parse_args()
    result = {
        "schema_version": 1,
        "data_root": str(args.data_root),
        "questions": audit_questions(args.data_root),
        "corpus": audit_corpus(args.data_root),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
