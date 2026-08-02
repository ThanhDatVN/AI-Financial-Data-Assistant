from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow.json as pajson
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vifinqa.indexing.manifest import iter_records, records_for_document  # noqa: E402
from vifinqa.parsing.document import statement_paths  # noqa: E402


def jsonl_to_parquet(jsonl_path: Path, parquet_path: Path) -> None:
    reader = pajson.open_json(
        jsonl_path, read_options=pajson.ReadOptions(block_size=8 * 1024 * 1024)
    )
    writer: pq.ParquetWriter | None = None
    try:
        for batch in reader:
            if writer is None:
                writer = pq.ParquetWriter(parquet_path, batch.schema, compression="zstd")
            writer.write_batch(batch)
    finally:
        if writer is not None:
            writer.close()


def _parse_job(job: tuple[str, str, int, str]) -> list[str]:
    raw_path, raw_root, first_table_id, table_ref_format = job
    records = records_for_document(
        Path(raw_path),
        data_root=Path(raw_root),
        first_table_id=first_table_id,
        table_ref_format=table_ref_format,
    )
    return [record.to_json() for record in records]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a provenance-preserving ViFinQA table manifest"
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/ViFinQA")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/table_manifest.jsonl")
    parser.add_argument(
        "--parquet", type=Path, default=ROOT / "data/processed/table_manifest.parquet"
    )
    parser.add_argument("--limit-documents", type=int)
    parser.add_argument("--limit-tables", type=int)
    parser.add_argument("--first-table-id", type=int, default=1)
    parser.add_argument("--table-ref-format", default="{doc_id}|table_{table_id}")
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--no-parquet", action="store_true")
    args = parser.parse_args()

    if not args.data_root.exists():
        raise SystemExit(f"Dataset not found: {args.data_root}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    count = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        if args.workers <= 1 or args.limit_tables is not None:
            for record in iter_records(
                args.data_root,
                first_table_id=args.first_table_id,
                table_ref_format=args.table_ref_format,
                limit_documents=args.limit_documents,
                limit_tables=args.limit_tables,
            ):
                handle.write(record.to_json() + "\n")
                count += 1
                if count % 1_000 == 0:
                    print(f"manifest tables: {count}", flush=True)
        else:
            paths = statement_paths(args.data_root)
            if args.limit_documents is not None:
                paths = paths[: args.limit_documents]
            jobs = [
                (
                    str(path),
                    str(args.data_root),
                    args.first_table_id,
                    args.table_ref_format,
                )
                for path in paths
            ]
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                for document_index, lines in enumerate(
                    executor.map(_parse_job, jobs, chunksize=1), start=1
                ):
                    for line in lines:
                        handle.write(line + "\n")
                    count += len(lines)
                    if document_index % 50 == 0 or document_index == len(jobs):
                        print(
                            f"manifest documents: {document_index}/{len(jobs)}; tables: {count}",
                            flush=True,
                        )
    if not args.no_parquet:
        jsonl_to_parquet(args.output, args.parquet)
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tables": count,
                "seconds": round(time.monotonic() - started, 3),
                "arguments": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "contract": {
                    "table_ref": args.table_ref_format,
                    "first_table_id": args.first_table_id,
                    "dashboard_verified": False,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {count} records to {args.output} and {args.parquet}")


if __name__ == "__main__":
    main()
