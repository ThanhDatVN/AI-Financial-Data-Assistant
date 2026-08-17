from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vifinqa.checkpoints.jsonl import (  # noqa: E402
    JsonlRowCheckpoint,
    write_json_atomic,
    write_jsonl_atomic,
)
from vifinqa.nlu.company import CompanyResolver  # noqa: E402
from vifinqa.nlu.query_spec import parse_query_spec  # noqa: E402
from vifinqa.retrieval.bm25 import BM25Index  # noqa: E402
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run metadata-routed hybrid retrieval")
    parser.add_argument(
        "--questions", type=Path, default=ROOT / "data/raw/ViFinQA/questions/questions.jsonl"
    )
    parser.add_argument("--companies", type=Path, default=ROOT / "data/raw/ViFinQA/code_stock.csv")
    parser.add_argument("--bm25", type=Path, default=ROOT / "data/index/bm25")
    parser.add_argument("--dense", type=Path)
    parser.add_argument("--dense-device")
    parser.add_argument("--project-revision")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/retrieval.jsonl")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Atomic per-question state; defaults beside --output",
    )
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
    # Imported here rather than at the top because a BM25-only ranking scored better than both
    # rankings built on the dense index (submissions 3135/3136/3137), and a run that does not
    # want the dense stage should not need faiss and sentence-transformers installed to skip it.
    dense = None
    if args.dense:
        from vifinqa.retrieval.dense import DenseIndex

        dense = DenseIndex.load(args.dense, device=args.dense_device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fingerprint: dict[str, object] = {
        "format_version": 1,
        "questions_sha256": _sha256(args.questions),
        "companies_sha256": _sha256(args.companies),
        "bm25_records_sha256": _sha256(args.bm25 / "records.jsonl"),
        "bm25_index_sha256": _sha256(args.bm25 / "data.csc.index.npy"),
        "dense_records_sha256": (_sha256(args.dense / "records.jsonl") if args.dense else None),
        "dense_index_sha256": (_sha256(args.dense / "index.faiss") if args.dense else None),
        "project_revision": args.project_revision,
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "question_ids": args.question_ids,
        "limit": args.limit,
    }
    checkpoint = JsonlRowCheckpoint(
        args.checkpoint_dir or args.output.with_name(args.output.name + ".checkpoint"),
        fingerprint=fingerprint,
    )
    completed = checkpoint.completed_ids()

    for row in questions:
        question_id = _as_int(row["id"], field="id")
        if question_id in completed:
            continue
        question = str(row["question"])
        spec = parse_query_spec(question, resolver)
        tickers = {entity.ticker for entity in spec.entities} or None
        years = {period.report_year for period in spec.periods} or None
        scope = next((entity.scope for entity in spec.entities if entity.scope), None)
        scopes = {scope, "unknown"} if scope else None
        route_tickers: list[str | None] = [*sorted(tickers)] if tickers else [None]
        route_years: list[int | None] = [*sorted(years)] if years else [None]
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
        hybrid_refs = [table_ref for table_ref, _ in fused]
        checkpoint.write(
            {
                "id": question_id,
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
                "hybrid": hybrid_refs,
                "fused": hybrid_refs,
            }
        )
    rows = checkpoint.consolidate(args.output)
    selected_ids = {_as_int(row["id"], field="id") for row in questions}
    output_rows = [row for row in rows if _as_int(row["id"], field="id") in selected_ids]
    if len(output_rows) != len(questions):
        raise ValueError(
            f"Checkpoint contains {len(output_rows)}/{len(questions)} selected retrieval rows"
        )
    if len(output_rows) != len(rows):
        write_jsonl_atomic(args.output, output_rows)
    # A ranking beside its own provenance, the way `34_merge_retrieval_shards.py` writes one.
    # A final run asserts this file exists before it starts, and nothing produced it here: the
    # only rankings that ever had one came through the reranker's merge step, so choosing the
    # cheaper and better-scoring BM25 ranking would have failed that assertion twelve hours in.
    write_json_atomic(
        args.output.with_suffix(args.output.suffix + ".metadata.json"),
        {
            **fingerprint,
            "rows": len(output_rows),
            "retrieval_sha256": _sha256(args.output),
            "ranking": "bm25" if dense is None else "hybrid",
        },
    )
    print(f"wrote {len(output_rows)} retrieval rows to {args.output}")


if __name__ == "__main__":
    main()
