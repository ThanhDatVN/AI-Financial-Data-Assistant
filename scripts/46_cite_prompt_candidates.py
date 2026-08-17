"""Cite exactly the tables the model was shown, to measure how often the gold one is among them.

`EXEC = coverage x P(gold table in the prompt) x P(correct | both)`. Two of those three are
measured; the middle one never has been, and it decides whether the next weeks go into retrieval
or into cell selection. The only number that ever stood in for it -- table recall 0.6283 at depth
20 -- came from a BM25-only ranking that no scored run ever used, so it is a lower bound on a
different list ([docs/12 section 9.1](../docs/12-kinh-nghiem.md)).

The dashboard will measure it directly if a submission cites the candidate set instead of the
answer's evidence: TABLES RECALL is then, per question, the share of gold tables the model could
possibly have read. One submission, no GPU, nothing regenerated.

The set has to be the one the generator actually built, not the top N of the ranking. It skips
candidates whose table parses to no number at all -- about one in sixteen -- so it reaches deeper
than N, and a cohort question is allowed more than N to begin with. Approximating both would
understate the very quantity being measured.

`answer`, `pandas_query` and `evidence` are untouched, so EXECUTION and ANSWER stay exactly as
they scored on the run this repackages, and `relevant_docs` is left alone too unless asked for.
Only the table citations change, which is the whole point: one number moves, and it is the one
being measured.

The references this writes are the manifest's own -- `{doc}|table_N` -- and the grader does not
read those. It wants `{doc}|{line_no}`, settled over five submissions and recorded in
[docs/12 section 1.1](../docs/12-kinh-nghiem.md). Submissions 3132 and 3133 were spent finding
that out again: every table metric came back 0.0 while EXECUTION and ANSWER were untouched, which
is what an unreadable citation list looks like rather than a low-recall one.

**Always follow this with `42_retarget_table_refs.py --grammar line` before packaging.**

`--candidate-tables` and `--scope-router` turn the same command into the rest of the experiment.
Depth says what a two-stage table selection could reach at best; the scope router says what
assuming the consolidated statement is worth, and both cost a submission and no GPU.
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
from vifinqa.evidence.store import TableStore, parsed_table_to_long_frame  # noqa: E402
from vifinqa.generation.prompt import numeric_cells_of  # noqa: E402
from vifinqa.indexing.manifest import ManifestRecord  # noqa: E402
from vifinqa.retrieval.routing import NO_ROUTING, scope_routed  # noqa: E402


def _routes(value: object) -> int:
    return len(value) if isinstance(value, list) else 0


_VARIABLE_RE = re.compile(r"^df(\d+)$")


def _rebuild_evidence(
    prediction: dict[str, object],
    *,
    shown: list[str],
    store: TableStore,
    root: Path,
    unit_source: str,
) -> int:
    """Write the CSVs this answer read, found by the position their variable names encode.

    The generator names its candidates df1, df2, ... in the order it built them, so `df8` is the
    eighth entry of `shown`. That is the same list this script just computed, which is why the
    mapping is sound here and nowhere else.
    """
    evidence = prediction.get("evidence")
    if not isinstance(evidence, list):
        return 0
    written = 0
    for item in evidence:
        if not isinstance(item, dict):
            continue
        match = _VARIABLE_RE.match(str(item.get("variable", "")))
        relative = str(item.get("csv_path", ""))
        if not match or not relative:
            raise ValueError(f"q{prediction.get('id')}: unreadable evidence entry {item}")
        index = int(match.group(1)) - 1
        if not 0 <= index < len(shown):
            raise ValueError(
                f"q{prediction.get('id')}: {item['variable']} is outside the {len(shown)} "
                "candidates this ranking and depth produce. The run being repackaged used a "
                "different --ranking-field or --candidate-tables."
            )
        destination = root / relative
        if destination.exists():
            continue
        record, table = store.load(shown[index], unit_source=unit_source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        parsed_table_to_long_frame(record, table).to_csv(destination, index=False, encoding="utf-8")
        written += 1
    return written


def _candidate_limit(spec: dict[str, object], *, minimum: int) -> int:
    """The generator's own rule: a cohort question is allowed one candidate per route."""
    tickers = max(1, _routes(spec.get("tickers")))
    years = max(1, _routes(spec.get("years")))
    return max(minimum, tickers * years)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path, help="A scored submission to repackage")
    parser.add_argument("--retrieval", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data/processed/table_manifest.parquet"
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/ViFinQA")
    parser.add_argument(
        "--candidate-tables",
        type=int,
        default=20,
        help="The generator's --candidate-tables for the run being repackaged",
    )
    parser.add_argument("--table-unit-source", default="latest")
    parser.add_argument(
        "--ranking-field",
        default="fused",
        help=(
            "Which ranking to read. `fused` is what the generator consumes, so it is the set the "
            "model saw. A reranked file also carries `hybrid`, the order before reranking, which "
            "makes the second experiment -- did reranking help or hurt -- a rerun of this "
            "command rather than another artefact to fetch."
        ),
    )
    parser.add_argument(
        "--scope-router",
        choices=("both", "consolidated", "separate"),
        default=NO_ROUTING,
        help=(
            "Assume this statement scope when the question does not say, and drop candidates "
            "from the other one, exactly as `50_generate_programs.py --scope-router` would. "
            "`both` reproduces the ranking every scored run has used"
        ),
    )
    parser.add_argument(
        "--usable-cache",
        type=Path,
        help=(
            "Remember which tables parse to a number. The verdict is a function of the corpus "
            "and the frozen manifest, so it is the same for every variant, and it is the whole "
            "cost of this script: an hour of parsing the first time, seconds afterwards"
        ),
    )
    parser.add_argument(
        "--write-evidence",
        type=Path,
        help=(
            "Also rebuild the evidence CSVs the submission cites, under this directory. The "
            "packager has to put each one in the ZIP, and an archived run drops them because "
            "they are a deterministic function of the corpus. `52_restore_evidence_csv.py` "
            "cannot rebuild them from a finalised submission -- it pairs evidence with "
            "`relevant_tables`, and finalising rewrote that list to the citation set. The "
            "mapping that still works is `dfN` -> the Nth candidate, and this script is the one "
            "that knows how the candidate list was built, so it is the one that can do it"
        ),
    )
    parser.add_argument(
        "--docs-from-tables",
        action="store_true",
        help=(
            "Also rewrite relevant_docs to the documents the cited tables sit in. Off by "
            "default: the run being repackaged names its documents from the question's own "
            "metadata and scores 0.9537 doing it, and changing two things at once would trade "
            "a known-good number for a second unknown. The point here is to read TABLES RECALL."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help=(
            "Stop after this many questions. For checking the script only -- the organiser "
            "discards a submission that is missing any question, so a limited run is never one."
        ),
    )
    args = parser.parse_args()

    manifest = pd.read_parquet(args.manifest)
    store = TableStore(
        args.data_root, [ManifestRecord.from_dict(row) for row in manifest.to_dict("records")]
    )
    rows = {
        int(row["id"]): row
        for row in (
            json.loads(line)
            for line in args.retrieval.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    predictions = json.loads(args.submission.read_text(encoding="utf-8"))
    if args.limit:
        predictions = predictions[: args.limit]

    # Candidate lists overlap heavily between questions about the same company and year, so
    # parsing each table once instead of once per appearance is the difference between minutes
    # and an hour. The verdict per table is a single bit and never changes within a run.
    has_numbers: dict[str, bool] = {}
    if args.usable_cache and args.usable_cache.exists():
        cached = json.loads(args.usable_cache.read_text(encoding="utf-8"))
        has_numbers = {str(ref): bool(value) for ref, value in cached.items()}
        print(f"loaded {len(has_numbers)} cached table verdicts from {args.usable_cache}")

    def usable(table_ref: str) -> str | None:
        if table_ref not in has_numbers:
            try:
                record, table = store.load(table_ref, unit_source=args.table_unit_source)
                frame = parsed_table_to_long_frame(record, table)
            except Exception:  # noqa: BLE001 - a table that will not load is one nobody saw
                has_numbers[table_ref] = False
            else:
                has_numbers[table_ref] = bool(numeric_cells_of(frame))
        return table_ref if has_numbers[table_ref] else None

    widths: list[int] = []
    rebuilt = 0
    for position, prediction in enumerate(predictions, start=1):
        row = rows[int(prediction["id"])]
        spec = row["query_spec"]
        limit = _candidate_limit(spec, minimum=args.candidate_tables)
        stated = spec.get("scope")
        ranking = scope_routed(
            row[args.ranking_field],
            records=store.records,
            scope=str(stated) if stated else None,
            policy=args.scope_router,
        )
        shown: list[str] = []
        for table_ref in ranking:
            if len(shown) >= limit:
                break
            # The generator fills the slot from further down rather than spending prompt budget
            # on a table with nothing to read, so the set reaches past `limit` in the ranking.
            kept = usable(str(table_ref))
            if kept is not None:
                shown.append(kept)
        if args.write_evidence:
            rebuilt += _rebuild_evidence(
                prediction,
                shown=shown,
                store=store,
                root=args.write_evidence,
                unit_source=args.table_unit_source,
            )
        prediction["relevant_tables"] = shown
        if args.docs_from_tables:
            prediction["relevant_docs"] = list(dict.fromkeys(ref.split("|", 1)[0] for ref in shown))
        widths.append(len(shown))
        if position % 100 == 0:
            print(f"  {position}/{len(predictions)}", flush=True)
            # The cold pass costs hours and the cache is the whole point of paying it once.
            # Written only at the end, a crash four hours in leaves nothing behind.
            if args.usable_cache:
                write_json_atomic(args.usable_cache, has_numbers)

    write_json_atomic(args.output, predictions)
    if args.usable_cache:
        write_json_atomic(args.usable_cache, has_numbers)
        print(f"  wrote {len(has_numbers)} table verdicts to {args.usable_cache}")
    if args.limit:
        print("  --limit was set: this file is a check, not a submission.")
    widths.sort()
    print(
        f"cited the {args.ranking_field} candidate set for {len(predictions)} questions "
        f"-> {args.output}"
    )
    print(f"  scope router: {args.scope_router}")
    if args.write_evidence:
        print(f"  rebuilt {rebuilt} evidence CSVs under {args.write_evidence}")
    print(
        f"  tables per question: min {widths[0]}, median {widths[len(widths) // 2]}, "
        f"max {widths[-1]}"
    )
    print("  TABLES RECALL on the dashboard is now P(gold table in the prompt).")


if __name__ == "__main__":
    main()
