"""Cite every table the routing can reach, to split "ranked too low" from "wrong document".

Table recall over the whole BM25 pool is 0.8613 (submission 3167), so 13.9% of gold tables are
outside everything the model could ever be shown. Document recall is 0.9585 on the same run, which
says the routing names the right reports -- but those two numbers are computed over different
things, and putting them side by side does not say where the 13.9% went.

Two candidates, and they point at opposite work:

* the ranking buried the table inside a document the routing did get right, in which case depth is
  the lever and a two-stage selection over 100 candidates is worth building;
* the routing never offered the document, in which case no depth helps, the pool already holds
  everything routing can give, and the remaining loss needs a different fix entirely.

This separates them. It cites every table of every document the question's own metadata can be
naming -- ticker x year x scope, filtered to what the corpus holds -- with no ranking involved at
all. TABLES RECALL is then the ceiling of the routing itself:

* near 0.95 => the ranking is losing about nine points, and depth is worth paying for;
* near 0.86 => the pool already reaches the routing's ceiling, and depth past 100 buys nothing.

`answer`, `pandas_query`, `evidence` and `relevant_docs` are untouched, so EXECUTION and ANSWER
come back exactly as they scored. Only the table citations change.

This is a diagnostic, not a candidate: it cites about 352 tables per question, so TABLES PRECISION
and F2 will be near zero by construction. That is the cost of reading one number.

**Always follow this with `42_retarget_table_refs.py --grammar line` before packaging.** The grader
reads `{doc}|{line_no}`; the manifest's own `{doc}|table_N` scores a flat 0.0 on every table column.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vifinqa.checkpoints.jsonl import write_json_atomic  # noqa: E402


def _routed_documents(spec: dict[str, object], policy: str, known: set[str]) -> list[str]:
    """Every report this question's own metadata can be naming, filtered to the corpus.

    The same construction `43_widen_citations.py` uses, because the point is to measure the
    routing this project actually ships rather than a second one written for the occasion.
    """
    tickers = [str(item) for item in (spec.get("tickers") or ())]
    years = [int(item) for item in (spec.get("years") or ())]
    stated = spec.get("scope")
    if stated:
        scopes = [str(stated)]
    elif policy == "both":
        scopes = ["consolidated", "separate"]
    else:
        scopes = [policy]
    named = (
        f"{ticker}_financial_statements_{year}_{scope}"
        for ticker in tickers
        for year in years
        for scope in scopes
    )
    return [document for document in dict.fromkeys(named) if document in known]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path, help="A scored submission to repackage")
    parser.add_argument("--retrieval", required=True, type=Path, help="Supplies each query_spec")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data/processed/table_manifest.parquet"
    )
    parser.add_argument(
        "--doc-router",
        choices=("both", "consolidated", "separate"),
        default="both",
        help=(
            "What to assume when the question does not state a scope. `both` is the honest "
            "ceiling of ticker-by-year routing, because it does not let a wrong scope guess be "
            "counted as a routing failure"
        ),
    )
    parser.add_argument(
        "--fallback-ranking-field",
        default="bm25",
        help=(
            "What to cite for the 20 questions whose metadata routes to nothing in the corpus. "
            "Leaving them empty would measure the routing's ceiling and the validator's patience "
            "at the same time"
        ),
    )
    args = parser.parse_args()

    manifest = pd.read_parquet(args.manifest, columns=["table_ref", "doc_id"])
    by_document: dict[str, list[str]] = defaultdict(list)
    for table_ref, doc_id in zip(manifest["table_ref"], manifest["doc_id"], strict=True):
        by_document[str(doc_id)].append(str(table_ref))
    known = set(by_document)

    rows = {
        int(row["id"]): row
        for row in (
            json.loads(line)
            for line in args.retrieval.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    predictions = json.loads(args.submission.read_text(encoding="utf-8"))

    widths: list[int] = []
    unrouted = 0
    for prediction in predictions:
        row = rows[int(prediction["id"])]
        documents = _routed_documents(row["query_spec"], args.doc_router, known)
        cited = [ref for document in documents for ref in by_document[document]]
        if not cited:
            unrouted += 1
            cited = [str(ref) for ref in row[args.fallback_ranking_field]]
        prediction["relevant_tables"] = cited
        widths.append(len(cited))

    write_json_atomic(args.output, predictions)
    widths.sort()
    print(f"cited every routed table for {len(predictions)} questions -> {args.output}")
    print(f"  doc router: {args.doc_router}")
    print(
        f"  tables per question: min {widths[0]}, median {widths[len(widths) // 2]}, "
        f"mean {statistics.mean(widths):.1f}, max {widths[-1]}"
    )
    print(f"  total references: {sum(widths):,}")
    print(f"  questions whose metadata routed nowhere: {unrouted} (cited the pool instead)")
    print("  TABLES RECALL on the dashboard is now the ceiling of the routing itself.")


if __name__ == "__main__":
    main()
