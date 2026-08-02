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
    parser.add_argument(
        "--checkpoint-size",
        type=int,
        default=256,
        help="Persist this many embeddings per resume-safe shard",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Store resume shards outside the final dense artifact directory",
    )
    parser.add_argument("--max-seq-length", type=int, default=8_192)
    parser.add_argument(
        "--max-batch-tokens",
        type=int,
        default=8_192,
        help="Adapt batch size so its longest padded batch stays within this token budget",
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--sort-by-length",
        action="store_true",
        help=(
            "Group similarly sized texts before sharding to reduce padding without "
            "truncating content"
        ),
    )
    parser.add_argument(
        "--device",
        action="append",
        help="Repeat to distribute encoding over multiple GPUs",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.final_run and not args.model_revision:
        parser.error("--final-run requires --model-revision with a pre-cutoff commit SHA")
    records = list(iter_manifest(args.manifest))
    if args.limit is not None:
        records = records[: args.limit]
    devices: str | list[str] | None = args.device
    if args.device and len(args.device) == 1:
        devices = args.device[0]
    index = DenseIndex.build_checkpointed(
        records,
        checkpoint_dir=args.checkpoint_dir or args.output / "checkpoints",
        model_id=args.model,
        model_revision=args.model_revision,
        batch_size=args.batch_size,
        checkpoint_size=args.checkpoint_size,
        max_seq_length=args.max_seq_length,
        max_batch_tokens=args.max_batch_tokens,
        sort_by_length=args.sort_by_length,
        device=devices,
        use_fp16=args.fp16,
    )
    index.save(args.output)
    print(f"wrote dense index with {len(records)} tables to {args.output}")


if __name__ == "__main__":
    main()
