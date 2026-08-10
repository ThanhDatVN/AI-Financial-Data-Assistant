"""Cite more of what retrieval already found, because F2 weights recall four times.

`relevant_tables` currently lists only the tables a program reads, which averages 1.14 per
question while retrieval had already surfaced twenty. Under F2 with beta=2 that is the wrong
trade: the first submission scored documents at precision 0.722 and recall 0.4701, and moving
recall up is worth several times what the matching precision costs.

Nothing here touches `answer`, `pandas_query` or `evidence`, so execution is unaffected and a
submission built this way still reproduces.
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

from vifinqa.checkpoints.jsonl import write_json_atomic  # noqa: E402


def _retrieved(path: Path) -> dict[int, list[str]]:
    ranked: dict[int, list[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            ranked[int(row["id"])] = [str(ref) for ref in row["fused"]]
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--retrieval", type=Path, default=ROOT / "outputs/retrieval.jsonl")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--tables",
        required=True,
        type=int,
        help="How many tables to cite per question, counting the ones the program read first",
    )
    parser.add_argument(
        "--docs",
        type=int,
        help=(
            "Cap the documents cited below what the tables span. Measured to be a mistake: "
            "cutting to one document took the document score from 0.7807 to 0.4521, so this "
            "exists only to reproduce that finding."
        ),
    )
    parser.add_argument(
        "--doc-depth",
        type=int,
        help=(
            "Draw documents from this many ranked tables rather than from the cited ones. The "
            "table score peaks near five citations while the document score was still climbing "
            "at 3.57 documents, so the two want different depths and only this direction is "
            "allowed: extra documents are legal, missing ones are not."
        ),
    )
    args = parser.parse_args()
    if args.tables < 1:
        parser.error("--tables must be at least 1")
    if args.docs is not None and args.docs < 1:
        parser.error("--docs must be at least 1")
    if args.doc_depth is not None:
        if args.doc_depth < args.tables:
            parser.error("--doc-depth must be at least --tables, or a cited table loses its doc")
        if args.docs is not None:
            parser.error("--docs and --doc-depth pull in opposite directions; pass one")

    ranked = _retrieved(args.retrieval)
    predictions = json.loads(args.submission.read_text(encoding="utf-8"))
    widened = 0
    for prediction in predictions:
        question_id = int(prediction["id"])
        # The tables the program read come first: they are the ones it can defend.
        cited = [str(ref) for ref in prediction["relevant_tables"]]
        for candidate in ranked.get(question_id, []):
            if len(cited) >= args.tables:
                break
            if candidate not in cited:
                cited.append(candidate)
        widened += len(cited) - len(prediction["relevant_tables"])
        prediction["relevant_tables"] = cited
        sources = cited
        if args.doc_depth is not None:
            sources = list(cited)
            for candidate in ranked.get(question_id, []):
                if len(sources) >= args.doc_depth:
                    break
                if candidate not in sources:
                    sources.append(candidate)
        documents = list(dict.fromkeys(ref.split("|", 1)[0] for ref in sources))
        prediction["relevant_docs"] = documents if args.docs is None else documents[: args.docs]
    write_json_atomic(args.output, predictions)
    tables_each = sum(len(row["relevant_tables"]) for row in predictions) / len(predictions)
    docs_each = sum(len(row["relevant_docs"]) for row in predictions) / len(predictions)
    print(
        f"added {widened} citations across {len(predictions)} questions; now "
        f"{tables_each:.2f} tables and {docs_each:.2f} documents per question -> {args.output}"
    )


if __name__ == "__main__":
    main()
