from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import TypedDict

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RANKINGS = ("bm25", "dense", "fused", "reranked")


class RankSource(TypedDict):
    run: str
    ranking: str
    rank: int


class PoolCandidate(TypedDict):
    table_ref: str
    sources: list[RankSource]
    relevance: str
    sufficiency: str
    notes: str


class PoolRow(TypedDict):
    schema_version: int
    id: int
    question: str
    candidates: list[PoolCandidate]


def _run_arg(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("run must use NAME=PATH")
    return name.strip(), Path(raw_path)


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _as_int(value: object, *, field: str) -> int:
    if not isinstance(value, int | str):
        raise TypeError(f"{field} must be an integer or string")
    return int(value)


def _merge_rankings(
    runs: list[tuple[str, list[dict[str, object]]]],
    *,
    selected_ids: set[int],
    depth: int,
    rankings: tuple[str, ...] = DEFAULT_RANKINGS,
) -> dict[int, PoolRow]:
    questions: dict[int, str] = {}
    sources: dict[int, dict[str, list[RankSource]]] = defaultdict(lambda: defaultdict(list))
    for run_name, rows in runs:
        for row in rows:
            question_id = _as_int(row["id"], field="id")
            if question_id not in selected_ids:
                continue
            question = str(row.get("question", ""))
            if question:
                previous = questions.setdefault(question_id, question)
                if previous != question:
                    raise ValueError(f"question text mismatch for id={question_id}")
            for ranking in rankings:
                values = row.get(ranking, [])
                if not isinstance(values, list):
                    raise TypeError(f"{run_name}:{ranking} must be a list for id={question_id}")
                seen_in_ranking: set[str] = set()
                for rank, raw_ref in enumerate(values[:depth], start=1):
                    table_ref = str(raw_ref)
                    if not table_ref or table_ref in seen_in_ranking:
                        continue
                    seen_in_ranking.add(table_ref)
                    sources[question_id][table_ref].append(
                        {"run": run_name, "ranking": ranking, "rank": rank}
                    )

    merged: dict[int, PoolRow] = {}
    for question_id in sorted(selected_ids):
        candidates: list[PoolCandidate] = []
        for table_ref, candidate_sources in sources[question_id].items():
            ordered_sources = sorted(
                candidate_sources,
                key=lambda item: (int(item["rank"]), str(item["run"]), str(item["ranking"])),
            )
            candidates.append(
                {
                    "table_ref": table_ref,
                    "sources": ordered_sources,
                    "relevance": "unjudged",
                    "sufficiency": "unknown",
                    "notes": "",
                }
            )
        candidates.sort(
            key=lambda item: (
                min(int(source["rank"]) for source in item["sources"]),
                -len(item["sources"]),
                sum(int(source["rank"]) for source in item["sources"]),
                str(item["table_ref"]),
            )
        )
        merged[question_id] = {
            "schema_version": 1,
            "id": question_id,
            "question": questions.get(question_id, ""),
            "candidates": candidates,
        }
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Pool candidates from diverse retrieval runs")
    parser.add_argument("--template", type=Path, default=ROOT / "annotations/qrels_template.jsonl")
    parser.add_argument(
        "--run",
        type=_run_arg,
        action="append",
        required=True,
        help="Retrieval run as NAME=PATH; repeat for diverse systems",
    )
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--output", type=Path, default=ROOT / "annotations/qrels_pool.jsonl")
    args = parser.parse_args()
    if args.depth <= 0:
        parser.error("--depth must be positive")
    run_names = [name for name, _ in args.run]
    if len(run_names) != len(set(run_names)):
        parser.error("--run names must be unique")

    template = _load_jsonl(args.template)
    template_by_id = {_as_int(row["id"], field="id"): row for row in template}
    if len(template_by_id) != len(template):
        raise ValueError("template contains duplicate question ids")
    loaded_runs = [(name, _load_jsonl(path)) for name, path in args.run]
    merged = _merge_rankings(
        loaded_runs,
        selected_ids=set(template_by_id),
        depth=args.depth,
    )
    for question_id, row in merged.items():
        template_question = str(template_by_id[question_id]["question"])
        if row["question"] and row["question"] != template_question:
            raise ValueError(f"template/run question mismatch for id={question_id}")
        row["question"] = template_question
        if not row["candidates"]:
            raise ValueError(f"no pooled candidates for id={question_id}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for question_id in sorted(merged):
            handle.write(json.dumps(merged[question_id], ensure_ascii=False) + "\n")
    candidate_count = sum(len(row["candidates"]) for row in merged.values())
    print(
        f"wrote {len(merged)} questions / {candidate_count} pooled candidates "
        f"from {len(loaded_runs)} runs to {args.output}"
    )


if __name__ == "__main__":
    main()
