from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vifinqa.evidence.store import TableStore  # noqa: E402


def _filename(table_ref: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", table_ref) + ".csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export retrieved raw tables as long evidence CSVs"
    )
    parser.add_argument("table_ref", nargs="+")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/ViFinQA")
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data/processed/table_manifest.parquet"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/evidence/data")
    args = parser.parse_args()
    refs = set(args.table_ref)
    store = (
        TableStore.from_parquet(args.data_root, args.manifest, refs)
        if args.manifest.suffix == ".parquet"
        else TableStore.from_manifest(args.data_root, args.manifest)
    )
    for table_ref in args.table_ref:
        output = store.export_csv(table_ref, args.output / _filename(table_ref))
        print(f"{table_ref}\t{output}")


if __name__ == "__main__":
    main()
