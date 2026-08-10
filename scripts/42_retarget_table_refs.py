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

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vifinqa.checkpoints.jsonl import write_json_atomic  # noqa: E402

_REF = re.compile(r"^(?P<doc>.+)\|table_(?P<ordinal>\d+)$")

GRAMMARS: dict[str, str] = {
    # What the companion repository uses, and what scored zero.
    "table_n": "{doc}|table_{ordinal}",
    # What the specification's own example looks like: a bare number after the pipe.
    "ordinal": "{doc}|{ordinal}",
    # The same, counted from zero, in case the example's number is an index.
    "ordinal0": "{doc}|{ordinal0}",
}


def retarget(reference: str, grammar: str) -> str:
    match = _REF.fullmatch(reference)
    if match is None:
        raise ValueError(f"Unrecognised table reference: {reference}")
    ordinal = int(match["ordinal"])
    return GRAMMARS[grammar].format(
        doc=match["doc"], ordinal=ordinal, ordinal0=ordinal - 1
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--grammar", required=True, choices=sorted(GRAMMARS))
    args = parser.parse_args()

    predictions = json.loads(args.submission.read_text(encoding="utf-8"))
    if not isinstance(predictions, list):
        raise ValueError("Submission root must be a JSON list")
    rewritten = 0
    for prediction in predictions:
        tables = prediction["relevant_tables"]
        prediction["relevant_tables"] = [retarget(str(ref), args.grammar) for ref in tables]
        rewritten += len(tables)
    write_json_atomic(args.output, predictions)
    print(
        f"rewrote {rewritten} references across {len(predictions)} questions "
        f"as {GRAMMARS[args.grammar]} -> {args.output}"
    )
    print("example:", predictions[0]["relevant_tables"][0])


if __name__ == "__main__":
    main()
