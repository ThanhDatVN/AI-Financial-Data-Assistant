"""Choose a fixed, fan-out-stratified subset for fast public-score experiments.

Every variant must run on exactly the same question IDs.  This script allocates a requested
sample proportionally across route fan-out strata (ticker count x year count), uses largest
remainders so the allocation sums exactly, and records the source hash for provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

STRATA = ("1", "2", "3", "4", "5", "6", "7", "8+")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise TypeError("Retrieval JSONL must contain objects")
    return rows


def route_fan_out(row: dict[str, Any]) -> int:
    spec = row.get("query_spec")
    if not isinstance(spec, dict):
        raise TypeError("Every retrieval row must contain query_spec")
    tickers = spec.get("tickers")
    years = spec.get("years")
    if not isinstance(tickers, list) or not isinstance(years, list):
        raise TypeError("query_spec.tickers and query_spec.years must be lists")
    return max(1, len(tickers)) * max(1, len(years))


def _stratum(fan_out: int) -> str:
    return "8+" if fan_out >= 8 else str(fan_out)


def proportional_allocation(counts: dict[str, int], size: int) -> dict[str, int]:
    """Allocate ``size`` items with Hamilton's largest-remainder method."""
    total = sum(counts.values())
    if size < 1 or size > total:
        raise ValueError(f"sample size must be in 1..{total}")
    exact = {name: size * counts.get(name, 0) / total for name in STRATA}
    allocated = {name: int(exact[name]) for name in STRATA}
    remaining = size - sum(allocated.values())
    order = sorted(
        STRATA,
        key=lambda name: (exact[name] - allocated[name], counts.get(name, 0)),
        reverse=True,
    )
    for name in order[:remaining]:
        allocated[name] += 1
    if any(allocated[name] > counts.get(name, 0) for name in STRATA):
        raise ValueError("A fan-out stratum is too small for its allocation")
    return allocated


def sample_ids(
    rows: list[dict[str, Any]], *, size: int, seed: int
) -> tuple[list[int], dict[str, int]]:
    pools: dict[str, list[int]] = defaultdict(list)
    seen: set[int] = set()
    for row in rows:
        question_id = int(row["id"])
        if question_id in seen:
            raise ValueError(f"Duplicate retrieval ID: {question_id}")
        seen.add(question_id)
        pools[_stratum(route_fan_out(row))].append(question_id)
    counts = {name: len(pools[name]) for name in STRATA}
    allocation = proportional_allocation(counts, size)
    rng = random.Random(seed)
    chosen: list[int] = []
    for name in STRATA:
        chosen.extend(rng.sample(sorted(pools[name]), allocation[name]))
    return sorted(chosen), allocation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("retrieval", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()

    rows = _load_jsonl(args.retrieval)
    ids, allocation = sample_ids(rows, size=args.size, seed=args.seed)
    counts: dict[str, int] = defaultdict(int)
    by_id = {int(row["id"]): row for row in rows}
    for question_id in ids:
        counts[_stratum(route_fan_out(by_id[question_id]))] += 1
    payload = {
        "seed": args.seed,
        "size": len(ids),
        "source": str(args.retrieval.as_posix()),
        "source_sha256": _sha256(args.retrieval),
        "strata": [
            {"fan_out": name, "selected": counts[name], "allocated": allocation[name]}
            for name in STRATA
        ],
        "ids": ids,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"selected={len(ids)} seed={args.seed} output={args.output}")
    print("strata=" + ", ".join(f"{name}:{counts[name]}" for name in STRATA))


if __name__ == "__main__":
    main()
