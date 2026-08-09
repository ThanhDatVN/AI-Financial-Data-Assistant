"""Rebuild the evidence CSVs a checkpoint refers to but does not carry.

The answers a run produced cannot be recreated, but the tables they cite can: the long
frames are a deterministic function of the corpus and the frozen manifest. Carrying them
between sessions costs gigabytes, rebuilding them costs minutes, so a checkpoint only has to
move the rows themselves.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vifinqa.evidence.store import TableStore, parsed_table_to_long_frame  # noqa: E402


def _required_frames(shards: list[Path]) -> dict[Path, str]:
    """Map every evidence CSV a completed row cites to the table it was written from."""
    required: dict[Path, str] = {}
    for shard in shards:
        for row_path in sorted((shard / "rows").glob("*.json")):
            row = json.loads(row_path.read_text(encoding="utf-8"))
            prediction = row.get("prediction")
            if not isinstance(prediction, dict):
                continue
            evidence = prediction.get("evidence")
            tables = prediction.get("relevant_tables")
            if not isinstance(evidence, list) or not isinstance(tables, list):
                continue
            if len(evidence) != len(tables):
                raise ValueError(f"Evidence and tables disagree in {row_path}")
            for item, table_ref in zip(evidence, tables, strict=True):
                if not isinstance(item, dict):
                    raise TypeError(f"Malformed evidence entry in {row_path}")
                csv_path = shard / str(item["csv_path"])
                if csv_path.parent.name != "data":
                    raise ValueError(f"Evidence path escapes data/: {item['csv_path']}")
                required[csv_path] = str(table_ref)
    return required


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", type=Path, nargs="+")
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data/processed/table_manifest.parquet"
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/ViFinQA")
    args = parser.parse_args()

    required = _required_frames(args.shards)
    missing = {path: ref for path, ref in required.items() if not path.is_file()}
    print(f"evidence CSVs cited {len(required)}, missing {len(missing)}")
    if not missing:
        return

    store = TableStore.from_parquet(args.data_root, args.manifest, set(missing.values()))
    for position, (path, table_ref) in enumerate(sorted(missing.items()), start=1):
        record, table = store.load(table_ref)
        path.parent.mkdir(parents=True, exist_ok=True)
        parsed_table_to_long_frame(record, table).to_csv(path, index=False, encoding="utf-8")
        if position % 500 == 0 or position == len(missing):
            print(f"rebuilt {position}/{len(missing)}")

    still_missing = [path for path in required if not path.is_file()]
    if still_missing:
        raise SystemExit(f"{len(still_missing)} evidence CSVs could not be rebuilt")
    print(f"restored {len(missing)} evidence CSVs")


if __name__ == "__main__":
    main()
