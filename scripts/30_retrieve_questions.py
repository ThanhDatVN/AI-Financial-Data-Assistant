from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vifinqa.nlu.company import CompanyResolver  # noqa: E402
from vifinqa.nlu.query_spec import parse_query_spec  # noqa: E402
from vifinqa.retrieval.bm25 import BM25Index  # noqa: E402
from vifinqa.retrieval.dense import DenseIndex  # noqa: E402
from vifinqa.retrieval.fusion import (  # noqa: E402
    balanced_round_robin,
    coverage_budget,
    reciprocal_rank_fusion,
)


def _load_questions(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _as_int(value: object, *, field: str) -> int:
    if not isinstance(value, int | str):
        raise TypeError(f"{field} must be an integer or string")
    return int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run metadata-routed hybrid retrieval")
    parser.add_argument(
        "--questions", type=Path, default=ROOT / "data/raw/ViFinQA/questions/questions.jsonl"
    )
    parser.add_argument("--companies", type=Path, default=ROOT / "data/raw/ViFinQA/code_stock.csv")
    parser.add_argument("--bm25", type=Path, default=ROOT / "data/index/bm25")
    parser.add_argument("--dense", type=Path)
    parser.add_argument("--dense-device")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/retrieval.jsonl")
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Minimum output candidates; route coverage can increase this value",
    )
    parser.add_argument("--candidate-k", type=int, default=2_000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--id", type=int, action="append", dest="question_ids")
    args = parser.parse_args()
    if args.top_k <= 0 or args.candidate_k <= 0:
        parser.error("--top-k and --candidate-k must be positive")

    questions = _load_questions(args.questions)
    if args.question_ids:
        selected = set(args.question_ids)
        questions = [row for row in questions if _as_int(row["id"], field="id") in selected]
    if args.limit is not None:
        questions = questions[: args.limit]
    resolver = CompanyResolver.from_csv(args.companies)
    bm25 = BM25Index.load(args.bm25)
    dense = DenseIndex.load(args.dense, device=args.dense_device) if args.dense else None
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in questions:
            question = str(row["question"])
            spec = parse_query_spec(question, resolver)
            tickers = {entity.ticker for entity in spec.entities} or None
            years = {period.report_year for period in spec.periods} or None
            scope = next((entity.scope for entity in spec.entities if entity.scope), None)
            scopes = {scope, "unknown"} if scope else None
            route_tickers: list[str | None] = []
            route_years: list[int | None] = []
            if tickers:
                route_tickers.extend(sorted(tickers))
            else:
                route_tickers.append(None)
            if years:
                route_years.extend(sorted(years))
            else:
                route_years.append(None)
            routes = [
                (route_ticker, route_year)
                for route_ticker in route_tickers
                for route_year in route_years
            ]
            output_k = coverage_budget(args.top_k, len(routes))
            route_k = max(1, math.ceil(output_k / len(routes)))
            bm25_routes: list[list[str]] = []
            for ticker, year in routes:
                augmented = question
                if ticker:
                    augmented += f"\nma co phieu: {ticker}"
                if year:
                    augmented += f"\nnam bao cao: {year}"
                bm25_hits = bm25.search(
                    augmented,
                    top_k=route_k,
                    candidate_k=args.candidate_k,
                    tickers={ticker} if ticker else None,
                    years={year} if year else None,
                    scopes=scopes,
                )
                bm25_routes.append([hit.table_ref for hit in bm25_hits])
            bm25_refs = balanced_round_robin(bm25_routes, limit=output_k)
            rankings = [bm25_refs]
            dense_refs: list[str] = []
            if dense:
                dense_routes: list[list[str]] = []
                for ticker, year in routes:
                    augmented = question
                    if ticker:
                        augmented += f"\nma co phieu: {ticker}"
                    if year:
                        augmented += f"\nnam bao cao: {year}"
                    dense_hits = dense.search(
                        augmented,
                        top_k=route_k,
                        candidate_k=args.candidate_k,
                        tickers={ticker} if ticker else None,
                        years={year} if year else None,
                        scopes=scopes,
                    )
                    dense_routes.append([hit.table_ref for hit in dense_hits])
                dense_refs = balanced_round_robin(dense_routes, limit=output_k)
                rankings.append(dense_refs)
            fused = reciprocal_rank_fusion(rankings)[:output_k]
            handle.write(
                json.dumps(
                    {
                        "id": _as_int(row["id"], field="id"),
                        "question": question,
                        "query_spec": {
                            "tickers": sorted(tickers) if tickers else [],
                            "years": sorted(years) if years else [],
                            "scope": scope,
                            "target_unit": spec.target_unit,
                            "target_divisor": spec.target_divisor,
                        },
                        "bm25": bm25_refs,
                        "dense": dense_refs,
                        "fused": [table_ref for table_ref, _ in fused],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"wrote {len(questions)} retrieval rows to {args.output}")


if __name__ == "__main__":
    main()
