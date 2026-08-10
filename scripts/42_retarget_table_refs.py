"""Rewrite the table-reference grammar of a finished submission.

The dashboard scored every table at zero while scoring documents at 0.48, and both come from
the same string: the part before the pipe was accepted and the part after it was not. Which
grammar the organiser expects has never been verifiable from the specification, whose only
example reads `doc|350` with no explanation of the number, so it has to be tested.

Rewriting is a string transform over a submission that already validates, so a candidate
costs a repackage rather than a run.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vifinqa.checkpoints.jsonl import write_json_atomic  # noqa: E402

_REF = re.compile(r"^(?P<doc>.+)\|table_(?P<ordinal>\d+)$")

# The specification calls the field "vị trí bảng trong báo cáo" and illustrates it with
# `AAA_financial_statements_2015_consolidated|350`. That example cannot be read literally: the
# question in it asks about a different company and year, its csv_path uses `table_1`, and the
# document it names holds 47 tables across 43 pages with a smallest character offset of 430, so
# 350 is none of those things. All that survives is the word "position", which every candidate
# below interprets differently.
GRAMMARS: dict[str, str] = {
    # G0: what the companion repository uses, and what scored zero.
    "table_n": "{doc}|table_{ordinal}",
    # G1: a bare number after the pipe, which is what the example resembles.
    "ordinal": "{doc}|{ordinal}",
    # G2: the same, counted from zero.
    "ordinal0": "{doc}|{ordinal0}",
    # G3: the prefix kept but the count starting at zero.
    "table_n0": "{doc}|table_{ordinal0}",
    # G4: position as the line the table starts on.
    "line": "{doc}|{line_no}",
    # G5: position as the page it appears on.
    "page": "{doc}|{page_no}",
    # G6: position across the whole corpus rather than within the report.
    "global": "{doc}|{global_index}",
}
# Which grammars need the manifest rather than only the reference string.
_NEEDS_MANIFEST = frozenset({"line", "page", "global"})


def _positions(manifest: Path) -> dict[str, dict[str, int]]:
    """Every positional field the manifest records, keyed by the reference we already emit."""
    frame = pd.read_parquet(
        manifest, columns=["table_ref", "doc_id", "table_id", "page_no", "line_no"]
    )
    frame = frame.sort_values(["doc_id", "table_id"]).reset_index(drop=True)
    frame["global_index"] = frame.index + 1
    return {
        str(row.table_ref): {
            "page_no": int(row.page_no) if row.page_no == row.page_no else -1,
            "line_no": int(row.line_no),
            "global_index": int(row.global_index),
        }
        for row in frame.itertuples(index=False)
    }


def retarget(
    reference: str, grammar: str, positions: dict[str, dict[str, int]] | None = None
) -> str:
    match = _REF.fullmatch(reference)
    if match is None:
        raise ValueError(f"Unrecognised table reference: {reference}")
    ordinal = int(match["ordinal"])
    fields = {"doc": match["doc"], "ordinal": ordinal, "ordinal0": ordinal - 1}
    if grammar in _NEEDS_MANIFEST:
        if positions is None or reference not in positions:
            raise ValueError(f"No manifest position recorded for {reference}")
        fields.update(positions[reference])
    return GRAMMARS[grammar].format(**fields)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--grammar", required=True, choices=sorted(GRAMMARS))
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data/processed/table_manifest.parquet"
    )
    args = parser.parse_args()
    positions = _positions(args.manifest) if args.grammar in _NEEDS_MANIFEST else None

    predictions = json.loads(args.submission.read_text(encoding="utf-8"))
    if not isinstance(predictions, list):
        raise ValueError("Submission root must be a JSON list")
    rewritten = 0
    for prediction in predictions:
        tables = prediction["relevant_tables"]
        prediction["relevant_tables"] = [
            retarget(str(ref), args.grammar, positions) for ref in tables
        ]
        rewritten += len(tables)
    write_json_atomic(args.output, predictions)
    print(
        f"rewrote {rewritten} references across {len(predictions)} questions "
        f"as {GRAMMARS[args.grammar]} -> {args.output}"
    )
    print("example:", predictions[0]["relevant_tables"][0])


if __name__ == "__main__":
    main()
