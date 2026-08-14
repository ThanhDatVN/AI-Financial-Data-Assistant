"""Apply answer conventions published by the organiser to a generated submission.

The organiser's reference prompt says that an undirected difference is an absolute,
non-negative value.  Generation can still emit an ordered subtraction, so this postprocessor
wraps the executable query in ``abs(...)`` and updates its declared answer in the same operation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vifinqa.submission.semantics import apply_absolute_difference_convention  # noqa: E402


def _load_submission(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise TypeError("Submission must be a JSON list of objects")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows, changed_ids = apply_absolute_difference_convention(_load_submission(args.submission))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"absolute differences applied={len(changed_ids)} ids={changed_ids}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
