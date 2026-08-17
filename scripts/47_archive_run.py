"""File a scored run away under a stable name, before the downloads folder is cleared.

The diagnostics of one session were already lost this way: they sat in Downloads under names
like `program_traces..txt`, the next download overwrote them, and the raw numbers behind
[RUN-031](../docs/nhat-ky-thi-nghiem.md) are gone. `outputs/` is outside git, so whatever this
writes is the only copy there will ever be.

It also parses the dashboard's ten columns, which is the one piece of reading in this project
that has been got wrong before and is settled in [docs/12 section 1.1](../docs/12-kinh-nghiem.md).
Encoding the order here means never reading it off by eye again:

    EXECUTION · TABLES F2 · DOCS F2 · T-P · T-R · T-MRR@5 · D-P · D-R · D-MRR@5 · ANSWER

What it keeps is what cannot be made again: the answers, and the trace that says how each one was
reached. The evidence CSVs inside a submission zip rebuild from the manifest in minutes and weigh
more than everything else combined, so they are dropped.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The dashboard's column order, settled over six submissions and not to be re-derived.
SCORE_COLUMNS = (
    "execution_accuracy",
    "tables_f2_macro",
    "docs_f2_macro",
    "tables_precision",
    "tables_recall",
    "tables_mrr5",
    "docs_precision",
    "docs_recall",
    "docs_mrr5",
    "answer_accuracy",
)

# What a run is worth keeping for, and the name it is kept under.
WANTED = {
    "program_traces": "program_traces.jsonl",
    "submission": "submission.json",
    "errors": "errors.jsonl",
    "error_attempts": "error_attempts.jsonl",
    "retrieval_hybrid_qc": "retrieval_hybrid_qc.json",
    "retrieval_reranked_qc": "retrieval_reranked_qc.json",
}


def _parse_scores(line: str) -> dict[str, float]:
    """Read the ten metrics out of a pasted dashboard row.

    The row also carries rank, team and timestamp, and the numbers are what matter, so take every
    decimal in order and insist there are exactly ten. Fewer or more means the row was pasted
    from a different view, and guessing which ten were meant is how a column gets misread.
    """
    numbers = [float(token) for token in re.findall(r"\d+\.\d+", line)]
    if len(numbers) != len(SCORE_COLUMNS):
        raise SystemExit(
            f"expected {len(SCORE_COLUMNS)} decimals in the score row, found {len(numbers)}: "
            f"{numbers}"
        )
    return dict(zip(SCORE_COLUMNS, numbers, strict=True))


def _harvest(source: Path, destination: Path) -> list[str]:
    """Copy what is worth keeping, under the name it is kept as."""
    kept: list[str] = []
    for path in sorted(source.rglob("*")):
        if path.is_dir():
            continue
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                for member in archive.namelist():
                    # The evidence CSVs rebuild from the manifest and dwarf everything else.
                    if member.startswith("data/") or not member.endswith(".json"):
                        continue
                    (destination / Path(member).name).write_bytes(archive.read(member))
                    kept.append(Path(member).name)
            continue
        stem = path.stem.rstrip(".")
        if stem in WANTED:
            shutil.copy(path, destination / WANTED[stem])
            kept.append(WANTED[stem])
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Folder holding what was downloaded")
    parser.add_argument("--submission-id", required=True, type=int)
    parser.add_argument(
        "--scores",
        required=True,
        help="The dashboard row, pasted whole; the ten decimals are read in column order",
    )
    parser.add_argument("--rank", type=int)
    parser.add_argument("--project-revision", default="")
    parser.add_argument("--note", default="")
    parser.add_argument("--runs", type=Path, default=ROOT / "outputs/runs")
    args = parser.parse_args()

    destination = args.runs / f"sub_{args.submission_id}"
    destination.mkdir(parents=True, exist_ok=True)
    kept = _harvest(args.source, destination)
    if not kept:
        raise SystemExit(f"nothing recognisable in {args.source}; expected {sorted(WANTED)}")

    scores = _parse_scores(args.scores)
    record: dict[str, object] = {
        "submission_id": args.submission_id,
        "rank": args.rank,
        "project_revision": args.project_revision,
        "note": args.note,
        **scores,
    }

    traces = destination / WANTED["program_traces"]
    if traces.is_file():
        rows = [
            json.loads(line) for line in traces.read_text(encoding="utf-8").splitlines() if line
        ]
        rescued = [row for row in rows if row.get("rescued")]
        fallback = [row for row in rows if row.get("fallback")]
        record["questions"] = len(rows)
        record["clean_programs"] = len(rows) - len(rescued) - len(fallback)
        record["rescued"] = len(rescued)
        record["fallback"] = len(fallback)
        record["measured_coverage"] = round((len(rows) - len(fallback)) / max(1, len(rows)), 4)
        by_class: dict[str, int] = {}
        for row in rescued:
            name = str(row.get("rescued_from") or "unrecorded")
            by_class[name] = by_class.get(name, 0) + 1
        record["rescued_from"] = dict(sorted(by_class.items(), key=lambda item: -item[1]))

    (destination / "scores.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"filed submission {args.submission_id} -> {destination}")
    for name in sorted(set(kept)):
        print(f"  {name}")
    print(f"  EXECUTION {scores['execution_accuracy']:.4f}", end="")
    if "measured_coverage" in record:
        print(f" = coverage {record['measured_coverage']} x precision on covered", end="")
    print()


if __name__ == "__main__":
    main()
