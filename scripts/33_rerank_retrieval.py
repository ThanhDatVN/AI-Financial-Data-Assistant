from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vifinqa.checkpoints.jsonl import JsonlRowCheckpoint  # noqa: E402
from vifinqa.indexing.manifest import ManifestRecord, iter_manifest  # noqa: E402
from vifinqa.retrieval.fusion import coverage_budget  # noqa: E402
from vifinqa.retrieval.rerank import (  # noqa: E402
    CrossEncoderReranker,
    preserve_route_coverage,
)


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
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise TypeError(f"{field} must be an integer or string")
    return int(value)


def _str_list(value: object, *, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{field} must be a list of strings")
    return list(value)


def _int_list(value: object, *, field: str) -> list[int]:
    if not isinstance(value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise TypeError(f"{field} must be a list of integers")
    return list(value)


def _select_rows(
    rows: list[dict[str, object]],
    *,
    question_ids: list[int] | None,
    limit: int | None,
    shard_count: int,
    shard_index: int,
) -> list[dict[str, object]]:
    ids = [_as_int(row.get("id"), field="id") for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Retrieval rows must have unique IDs")
    selected = rows
    if question_ids:
        by_id = dict(zip(ids, rows, strict=True))
        missing = sorted(set(question_ids) - set(by_id))
        if missing:
            raise ValueError(f"Unknown question IDs: {missing}")
        selected = [by_id[question_id] for question_id in question_ids]
    if limit is not None:
        selected = selected[:limit]
    return [row for position, row in enumerate(selected) if position % shard_count == shard_index]


def _candidate_refs(row: dict[str, object], *, limit: int) -> list[str]:
    raw = row.get("hybrid", row.get("fused"))
    refs = _str_list(raw, field="hybrid/fused")
    return list(dict.fromkeys(refs))[:limit]


def _load_candidate_records(
    records_path: Path,
    rows: list[dict[str, object]],
    *,
    candidate_tables: int,
) -> dict[str, ManifestRecord]:
    required = {
        table_ref for row in rows for table_ref in _candidate_refs(row, limit=candidate_tables)
    }
    records: dict[str, ManifestRecord] = {}
    for record in iter_manifest(records_path):
        if record.table_ref in required:
            records[record.table_ref] = record
    missing = sorted(required - set(records))
    if missing:
        raise ValueError(f"Records file is missing {len(missing)} candidates: {missing[:5]}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-encode and route-constrain hybrid retrieval"
    )
    parser.add_argument("--retrieval", required=True, type=Path)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Shard output directory")
    parser.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--model-revision")
    parser.add_argument("--project-revision")
    parser.add_argument("--final-run", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--max-length", type=int, default=8_192)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--max-batch-tokens",
        type=int,
        default=8_192,
        help="Reduce only batch size for long pairs; never shorten model context",
    )
    parser.add_argument("--candidate-tables", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--id", type=int, action="append", dest="question_ids")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if (
        min(
            args.max_length,
            args.batch_size,
            args.max_batch_tokens,
            args.candidate_tables,
            args.top_k,
        )
        <= 0
    ):
        parser.error("length, batch, candidate, and top-k values must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be in [0, --shard-count)")
    if args.final_run and (
        not args.model_revision or re.fullmatch(r"[0-9a-f]{40}", args.model_revision) is None
    ):
        parser.error("--final-run requires a lowercase 40-character model revision")
    if args.final_run and (
        not args.project_revision or re.fullmatch(r"[0-9a-f]{40}", args.project_revision) is None
    ):
        parser.error("--final-run requires the exact lowercase project Git revision")

    rows = _select_rows(
        _load_jsonl(args.retrieval),
        question_ids=args.question_ids,
        limit=args.limit,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )
    fingerprint: dict[str, object] = {
        "format_version": 1,
        "retrieval_sha256": _sha256(args.retrieval),
        "records_sha256": _sha256(args.records),
        "model": args.model,
        "model_revision": args.model_revision,
        "project_revision": args.project_revision,
        "max_length": args.max_length,
        "use_fp16": args.fp16,
        "batch_size": args.batch_size,
        "max_batch_tokens": args.max_batch_tokens,
        "candidate_tables": args.candidate_tables,
        "top_k": args.top_k,
        "question_ids": args.question_ids,
        "limit": args.limit,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
    }
    checkpoint = JsonlRowCheckpoint(args.output, fingerprint=fingerprint)
    completed = checkpoint.completed_ids()
    pending = [row for row in rows if _as_int(row.get("id"), field="id") not in completed]
    if pending:
        records = _load_candidate_records(
            args.records,
            pending,
            candidate_tables=args.candidate_tables,
        )
        reranker = CrossEncoderReranker(
            args.model,
            model_revision=args.model_revision,
            device=args.device,
            max_length=args.max_length,
            use_fp16=args.fp16,
        )
        started_at = time.monotonic()
        for position, row in enumerate(pending, start=1):
            refs = _candidate_refs(row, limit=args.candidate_tables)
            candidates = [records[table_ref] for table_ref in refs]
            hits = reranker.rerank(
                str(row.get("question", "")),
                candidates,
                top_k=len(candidates),
                batch_size=args.batch_size,
                max_batch_tokens=args.max_batch_tokens,
            )
            spec = row.get("query_spec")
            if not isinstance(spec, dict):
                raise TypeError("query_spec must be an object")
            tickers = _str_list(spec.get("tickers"), field="query_spec.tickers")
            years = _int_list(spec.get("years"), field="query_spec.years")
            scope = spec.get("scope")
            if scope is not None and not isinstance(scope, str):
                raise TypeError("query_spec.scope must be a string or null")
            route_tickers: list[str | None] = [*tickers] if tickers else [None]
            route_years: list[int | None] = [*years] if years else [None]
            routes = [(ticker, year) for ticker in route_tickers for year in route_years]
            output_k = coverage_budget(args.top_k, len(routes))
            covered = preserve_route_coverage(
                hits,
                records,
                routes=routes,
                scopes={scope, "unknown"} if scope else None,
                top_k=output_k,
            )
            output = dict(row)
            output["hybrid"] = refs
            output["reranked"] = [hit.table_ref for hit in covered]
            output["reranker_scores"] = [
                {"table_ref": hit.table_ref, "score": hit.score} for hit in hits
            ]
            output["fused"] = output["reranked"]
            checkpoint.write(output)
            if position == 1 or position % 10 == 0 or position == len(pending):
                elapsed = max(time.monotonic() - started_at, 1e-9)
                print(
                    f"shard {args.shard_index}: {position}/{len(pending)} new rows; "
                    f"{position / elapsed:.2f} questions/s"
                )

    output_rows = checkpoint.consolidate(args.output / "retrieval.jsonl")
    selected_ids = {_as_int(row.get("id"), field="id") for row in rows}
    if {_as_int(row.get("id"), field="id") for row in output_rows} != selected_ids:
        raise ValueError("Reranker checkpoint does not match the selected shard")
    print(f"completed reranker shard {args.shard_index}: {len(output_rows)} rows")


if __name__ == "__main__":
    main()
