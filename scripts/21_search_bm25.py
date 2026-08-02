from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vifinqa.retrieval.bm25 import BM25Index  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the local BM25 table index")
    parser.add_argument("query")
    parser.add_argument("--index", type=Path, default=ROOT / "data/index/bm25")
    parser.add_argument("--ticker", action="append")
    parser.add_argument("--year", type=int, action="append")
    parser.add_argument("--scope", choices=["consolidated", "separate", "aggregated", "unknown"])
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    index = BM25Index.load(args.index)
    hits = index.search(
        args.query,
        top_k=args.top_k,
        tickers=set(args.ticker) if args.ticker else None,
        years=set(args.year) if args.year else None,
        scope=args.scope,
    )
    print(json.dumps([asdict(hit) for hit in hits], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
