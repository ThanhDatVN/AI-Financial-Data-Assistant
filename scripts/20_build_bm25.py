from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vifinqa.indexing.manifest import iter_manifest  # noqa: E402
from vifinqa.retrieval.bm25 import BM25Index  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local accent-tolerant BM25 table index")
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data/processed/table_manifest.jsonl"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data/index/bm25")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    records = list(iter_manifest(args.manifest))
    if args.limit is not None:
        records = records[: args.limit]
    index = BM25Index.build(records)
    index.save(args.output)
    print(f"wrote BM25 index with {len(records)} tables to {args.output}")


if __name__ == "__main__":
    main()
