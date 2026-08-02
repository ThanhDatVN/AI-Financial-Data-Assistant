from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vifinqa.checkpoints.jsonl import write_json_atomic, write_jsonl_atomic  # noqa: E402


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _row_id(row: dict[str, object]) -> int:
    value = row.get("id")
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise TypeError("id must be an integer or string")
    return int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge disjoint reranker shards")
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-rows", type=int)
    args = parser.parse_args()
    if args.expected_rows is not None and args.expected_rows <= 0:
        parser.error("--expected-rows must be positive")

    metadata_rows = [
        json.loads((shard / "run_metadata.json").read_text(encoding="utf-8"))
        for shard in args.shards
    ]
    identities: list[dict[str, object]] = []
    shard_indices: list[int] = []
    shard_counts: list[int] = []
    for metadata in metadata_rows:
        identity = dict(metadata)
        shard_indices.append(int(identity.pop("shard_index")))
        shard_counts.append(int(identity.pop("shard_count")))
        identities.append(identity)
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("Reranker shards have different run fingerprints")
    if sorted(shard_indices) != list(range(len(args.shards))):
        raise ValueError(f"Expected shard indices 0..{len(args.shards) - 1}: {shard_indices}")
    if any(count != len(args.shards) for count in shard_counts):
        raise ValueError("Shard-count metadata does not match the supplied shards")

    rows = [row for shard in args.shards for row in _load_jsonl(shard / "retrieval.jsonl")]
    ids = [_row_id(row) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question IDs across reranker shards")
    rows.sort(key=_row_id)
    if args.expected_rows is not None and len(rows) != args.expected_rows:
        raise ValueError(f"Merged {len(rows)} rows; expected {args.expected_rows}")
    for row in rows:
        reranked = row.get("reranked")
        if not isinstance(reranked, list) or not reranked or row.get("fused") != reranked:
            raise ValueError(f"Invalid reranked output for id={_row_id(row)}")

    write_jsonl_atomic(args.output, rows)
    merged_metadata = dict(identities[0])
    merged_metadata["merged_shards"] = len(args.shards)
    write_json_atomic(
        args.output.with_suffix(args.output.suffix + ".metadata.json"), merged_metadata
    )
    print(f"merged reranker shards={len(args.shards)} rows={len(rows)} to {args.output}")


if __name__ == "__main__":
    main()
