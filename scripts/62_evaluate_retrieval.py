from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vifinqa.checkpoints.jsonl import write_json_atomic  # noqa: E402
from vifinqa.eval.metrics import retrieval_metrics  # noqa: E402


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _row_id(row: dict[str, object]) -> int:
    value = row.get("id")
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise TypeError("id must be an integer or string")
    return int(value)


def _strings(value: object, *, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{field} must be a list of strings")
    return list(value)


def evaluate(
    qrels: list[dict[str, object]],
    run: list[dict[str, object]],
    *,
    rankings: list[str],
    cutoffs: list[int],
    statuses: set[str],
) -> dict[str, object]:
    gold_rows = [
        row
        for row in qrels
        if row.get("status") in statuses
        and row.get("answerability") == "ANSWERABLE"
        and row.get("gold_tables")
    ]
    if not gold_rows:
        raise ValueError("No reviewed/adjudicated answerable qrels with gold_tables")
    gold_by_id = {_row_id(row): row for row in gold_rows}
    if len(gold_by_id) != len(gold_rows):
        raise ValueError("Qrels contain duplicate IDs")
    run_by_id = {_row_id(row): row for row in run}
    if len(run_by_id) != len(run):
        raise ValueError("Retrieval run contains duplicate IDs")
    missing = sorted(set(gold_by_id) - set(run_by_id))
    if missing:
        raise ValueError(f"Retrieval run is missing {len(missing)} judged IDs: {missing[:10]}")

    results: dict[str, object] = {}
    slices: dict[str, list[int]] = defaultdict(list)
    for question_id, qrel in gold_by_id.items():
        slices[str(qrel.get("stratum", "unknown"))].append(question_id)
    for ranking in rankings:
        ranking_results: dict[str, object] = {}
        for cutoff in cutoffs:
            per_question: list[dict[str, float | bool]] = []
            for question_id, qrel in gold_by_id.items():
                retrieved = _strings(
                    run_by_id[question_id].get(ranking, []),
                    field=f"{ranking}[id={question_id}]",
                )[:cutoff]
                gold = set(_strings(qrel.get("gold_tables"), field="gold_tables"))
                metrics = retrieval_metrics(retrieved, gold)
                per_question.append(
                    {
                        "precision": metrics.precision,
                        "recall": metrics.recall,
                        "f2": metrics.f2,
                        "hit": bool(set(retrieved) & gold),
                        "complete": gold.issubset(retrieved),
                    }
                )
            ranking_results[str(cutoff)] = {
                "questions": len(per_question),
                "precision": statistics.fmean(float(row["precision"]) for row in per_question),
                "recall": statistics.fmean(float(row["recall"]) for row in per_question),
                "f2": statistics.fmean(float(row["f2"]) for row in per_question),
                "hit_rate": statistics.fmean(bool(row["hit"]) for row in per_question),
                "complete_evidence_rate": statistics.fmean(
                    bool(row["complete"]) for row in per_question
                ),
            }
        results[ranking] = ranking_results
    return {
        "judged_questions": len(gold_by_id),
        "accepted_statuses": sorted(statuses),
        "slice_counts": {name: len(ids) for name, ids in sorted(slices.items())},
        "rankings": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval against reviewed table qrels")
    parser.add_argument("--qrels", required=True, type=Path)
    parser.add_argument("--retrieval", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--ranking",
        action="append",
        dest="rankings",
        default=None,
        help="Ranking field; repeat as needed (default: bm25,dense,hybrid,reranked,fused)",
    )
    parser.add_argument("--cutoff", action="append", type=int, dest="cutoffs", default=None)
    parser.add_argument("--status", action="append", dest="statuses", default=None)
    args = parser.parse_args()
    rankings = args.rankings or ["bm25", "dense", "hybrid", "reranked", "fused"]
    cutoffs = sorted(set(args.cutoffs or [1, 5, 10, 20]))
    if not rankings or any(cutoff <= 0 for cutoff in cutoffs):
        parser.error("rankings must be non-empty and cutoffs must be positive")
    report = evaluate(
        _load_jsonl(args.qrels),
        _load_jsonl(args.retrieval),
        rankings=rankings,
        cutoffs=cutoffs,
        statuses=set(args.statuses or ["reviewed", "adjudicated"]),
    )
    write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
