from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vifinqa.indexing.manifest import iter_manifest  # noqa: E402
from vifinqa.retrieval.dense import DenseIndex  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a normalized FAISS dense table index")
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data/processed/table_manifest.jsonl"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data/index/bge_m3")
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--model-revision")
    parser.add_argument(
        "--final-run",
        action="store_true",
        help="Refuse an unpinned model revision for a submission-candidate index",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.final_run and not args.model_revision:
        parser.error("--final-run requires --model-revision with a pre-cutoff commit SHA")
    records = list(iter_manifest(args.manifest))
    if args.limit is not None:
        records = records[: args.limit]
    index = DenseIndex.build(
        records,
        model_id=args.model,
        model_revision=args.model_revision,
        batch_size=args.batch_size,
        device=args.device,
    )
    index.save(args.output)
    print(f"wrote dense index with {len(records)} tables to {args.output}")


if __name__ == "__main__":
    main()
