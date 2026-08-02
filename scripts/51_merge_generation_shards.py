from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _as_id(row: dict[str, object]) -> int:
    value = row.get("id")
    if not isinstance(value, int | str):
        raise TypeError("Every shard row must contain an integer ID")
    return int(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_unique(rows: list[dict[str, object]], *, label: str) -> None:
    ids = [_as_id(row) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs across {label} shards")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge disjoint generation shards")
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
    identities = []
    shard_indices = []
    shard_counts = []
    for metadata in metadata_rows:
        identity = dict(metadata)
        shard_indices.append(int(identity.pop("shard_index")))
        shard_counts.append(int(identity.pop("shard_count")))
        identities.append(identity)
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("Generation shards have different run fingerprints")
    if sorted(shard_indices) != list(range(len(args.shards))):
        raise ValueError(f"Expected shard indices 0..{len(args.shards) - 1}: {shard_indices}")
    if any(count != len(args.shards) for count in shard_counts):
        raise ValueError(f"Shard count metadata does not match {len(args.shards)}: {shard_counts}")

    predictions = [row for shard in args.shards for row in _load_jsonl(shard / "predictions.jsonl")]
    errors = [row for shard in args.shards for row in _load_jsonl(shard / "errors.jsonl")]
    attempts = [row for shard in args.shards for row in _load_jsonl(shard / "error_attempts.jsonl")]
    traces = [row for shard in args.shards for row in _load_jsonl(shard / "program_traces.jsonl")]
    for label, rows in (("prediction", predictions), ("error", errors), ("trace", traces)):
        _assert_unique(rows, label=label)
        rows.sort(key=_as_id)
    prediction_ids = {_as_id(row) for row in predictions}
    error_ids = {_as_id(row) for row in errors}
    trace_ids = {_as_id(row) for row in traces}
    if prediction_ids & error_ids:
        raise ValueError("A question cannot be both completed and unresolved")
    if trace_ids != prediction_ids:
        raise ValueError("Every merged prediction must have exactly one trace")
    if args.expected_rows is not None and len(prediction_ids | error_ids) != args.expected_rows:
        raise ValueError(
            f"Merged {len(prediction_ids | error_ids)} rows; expected {args.expected_rows}"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    data_output = args.output / "data"
    data_output.mkdir(exist_ok=True)
    for shard in args.shards:
        for source in (shard / "data").glob("*.csv"):
            destination = data_output / source.name
            if destination.exists() and _sha256(destination) != _sha256(source):
                raise ValueError(f"Conflicting evidence CSV across shards: {source.name}")
            if not destination.exists():
                shutil.copy2(source, destination)

    _write_jsonl(args.output / "predictions.jsonl", predictions)
    _write_jsonl(args.output / "errors.jsonl", errors)
    _write_jsonl(args.output / "error_attempts.jsonl", attempts)
    _write_jsonl(args.output / "program_traces.jsonl", traces)
    merged_metadata = dict(identities[0])
    merged_metadata["merged_shards"] = len(args.shards)
    (args.output / "run_metadata.json").write_text(
        json.dumps(merged_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output / "submission.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"merged shards={len(args.shards)} predictions={len(predictions)} "
        f"errors={len(errors)} output={args.output}"
    )


if __name__ == "__main__":
    main()
