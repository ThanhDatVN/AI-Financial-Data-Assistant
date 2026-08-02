from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class Availability:
    exact: set[tuple[str, int, str]]
    ticker_year: set[tuple[str, int]]
    ticker_scope: set[tuple[str, str]]
    year_scope: set[tuple[int, str]]
    tickers: set[str]
    years: set[int]
    scopes: set[str]

    @classmethod
    def from_metadata(cls, items: set[tuple[str, int, str]]) -> Availability:
        return cls(
            exact=items,
            ticker_year={(ticker, year) for ticker, year, _ in items},
            ticker_scope={(ticker, scope) for ticker, _, scope in items},
            year_scope={(year, scope) for _, year, scope in items},
            tickers={ticker for ticker, _, _ in items},
            years={year for _, year, _ in items},
            scopes={scope for _, _, scope in items},
        )

    def contains(self, *, ticker: str | None, year: int | None, scope: str | None) -> bool:
        allowed_scopes = (scope, "unknown") if scope else ()
        if ticker is not None and year is not None:
            if scope:
                return any(
                    (ticker, year, item_scope) in self.exact for item_scope in allowed_scopes
                )
            return (ticker, year) in self.ticker_year
        if ticker is not None:
            if scope:
                return any(
                    (ticker, item_scope) in self.ticker_scope for item_scope in allowed_scopes
                )
            return ticker in self.tickers
        if year is not None:
            if scope:
                return any((year, item_scope) in self.year_scope for item_scope in allowed_scopes)
            return year in self.years
        if scope:
            return any(item_scope in self.scopes for item_scope in allowed_scopes)
        return bool(self.exact)


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_int(value: object, *, field: str) -> int:
    if not isinstance(value, int | str):
        raise TypeError(f"{field} must be an integer or string")
    return int(value)


def _str_list(value: object, *, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{field} must be a list of strings")
    return value


def _int_list(value: object, *, field: str) -> list[int]:
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise TypeError(f"{field} must be a list of integers")
    return value


def _route_matches(
    metadata: tuple[str, int, str],
    *,
    ticker: str | None,
    year: int | None,
    scope: str | None,
) -> bool:
    actual_ticker, actual_year, actual_scope = metadata
    return (
        (ticker is None or actual_ticker == ticker)
        and (year is None or actual_year == year)
        and (scope is None or actual_scope in {scope, "unknown"})
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed retrieval artefact validation")
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "data/raw/ViFinQA/questions/questions.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/processed/table_manifest.parquet",
    )
    parser.add_argument("--retrieval", type=Path, default=ROOT / "outputs/retrieval.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/retrieval_qc.json")
    args = parser.parse_args()

    questions = _load_jsonl(args.questions)
    rows = _load_jsonl(args.retrieval)
    manifest = pq.read_table(
        args.manifest,
        columns=["table_ref", "ticker", "report_year", "scope"],
    )
    metadata_by_ref = {
        str(item["table_ref"]): (
            str(item["ticker"]),
            int(item["report_year"]),
            str(item["scope"]),
        )
        for item in manifest.to_pylist()
    }
    availability = Availability.from_metadata(set(metadata_by_ref.values()))
    question_by_id = {_as_int(item["id"], field="id"): str(item["question"]) for item in questions}

    failures: dict[str, list[object]] = {
        "row_or_question": [],
        "empty": [],
        "duplicate_refs": [],
        "unknown_refs": [],
        "wrong_metadata": [],
        "available_routes_missing": [],
    }
    unavailable_routes: list[dict[str, object]] = []
    short_candidates: list[dict[str, int]] = []
    seen_ids: list[int] = []
    multi_entity = 0
    max_routes = 0
    max_candidates = 0

    for row in rows:
        question_id = _as_int(row["id"], field="id")
        seen_ids.append(question_id)
        if question_by_id.get(question_id) != str(row.get("question", "")):
            failures["row_or_question"].append(question_id)
        spec = row.get("query_spec")
        if not isinstance(spec, dict):
            failures["row_or_question"].append({"id": question_id, "query_spec": "invalid"})
            continue
        tickers = _str_list(spec.get("tickers"), field="query_spec.tickers")
        years = _int_list(spec.get("years"), field="query_spec.years")
        raw_scope = spec.get("scope")
        if raw_scope is not None and not isinstance(raw_scope, str):
            raise TypeError("query_spec.scope must be a string or null")
        scope = raw_scope
        fused = _str_list(row.get("fused"), field="fused")
        if len(tickers) > 1:
            multi_entity += 1
        if not fused:
            failures["empty"].append(question_id)
        if len(fused) != len(set(fused)):
            failures["duplicate_refs"].append(question_id)
        if len(fused) < 20:
            short_candidates.append({"id": question_id, "candidates": len(fused)})
        max_candidates = max(max_candidates, len(fused))

        known_fused: list[tuple[str, tuple[str, int, str]]] = []
        for table_ref in fused:
            metadata = metadata_by_ref.get(table_ref)
            if metadata is None:
                failures["unknown_refs"].append({"id": question_id, "table_ref": table_ref})
                continue
            known_fused.append((table_ref, metadata))
            if (
                (tickers and metadata[0] not in tickers)
                or (years and metadata[1] not in years)
                or (scope is not None and metadata[2] not in {scope, "unknown"})
            ):
                failures["wrong_metadata"].append(
                    {"id": question_id, "table_ref": table_ref, "metadata": metadata}
                )

        route_tickers: list[str | None] = []
        route_years: list[int | None] = []
        if tickers:
            route_tickers.extend(tickers)
        else:
            route_tickers.append(None)
        if years:
            route_years.extend(years)
        else:
            route_years.append(None)
        routes = [(ticker, year) for ticker in route_tickers for year in route_years]
        max_routes = max(max_routes, len(routes))
        for ticker, year in routes:
            available = availability.contains(ticker=ticker, year=year, scope=scope)
            hit = any(
                _route_matches(item, ticker=ticker, year=year, scope=scope)
                for _, item in known_fused
            )
            route: dict[str, object] = {
                "id": question_id,
                "ticker": ticker,
                "year": year,
                "scope": scope,
            }
            if available and not hit:
                failures["available_routes_missing"].append(route)
            elif not available:
                unavailable_routes.append(route)

    expected_ids = [_as_int(item["id"], field="id") for item in questions]
    if seen_ids != expected_ids or len(seen_ids) != len(set(seen_ids)):
        failures["row_or_question"].append(
            {
                "expected_ids": len(expected_ids),
                "actual_ids": len(seen_ids),
                "ordered_unique": False,
            }
        )
    failure_counts = {name: len(items) for name, items in failures.items()}
    report: dict[str, object] = {
        "schema_version": 1,
        "retrieval_sha256": _sha256(args.retrieval),
        "manifest_sha256": _sha256(args.manifest),
        "rows": len(rows),
        "manifest_refs": len(metadata_by_ref),
        "multi_entity_questions": multi_entity,
        "max_routes_per_question": max_routes,
        "max_candidates_per_question": max_candidates,
        "short_candidate_questions": short_candidates,
        "unavailable_routes": {
            "count": len(unavailable_routes),
            "examples": unavailable_routes[:100],
        },
        "failure_counts": failure_counts,
        "failure_examples": {name: items[:100] for name, items in failures.items()},
        "passed": not any(failure_counts.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "failure_examples"}))
    if not report["passed"]:
        raise SystemExit(f"retrieval QC failed; see {args.output}")


if __name__ == "__main__":
    main()
