from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vifinqa.nlu.company import CompanyResolver  # noqa: E402
from vifinqa.nlu.query_spec import parse_query_spec  # noqa: E402
from vifinqa.parsing.normalize import ascii_words  # noqa: E402


def _band(count: int) -> str:
    return "none" if count == 0 else "one" if count == 1 else "multi"


def _row_id(row: dict[str, object]) -> int:
    value = row["id"]
    if not isinstance(value, int | str):
        raise TypeError("question id must be an integer or string")
    return int(value)


def _sample_round_robin(
    groups: dict[str, list[dict[str, object]]], *, size: int, seed: int
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    queues: dict[str, deque[dict[str, object]]] = {}
    for name, rows in groups.items():
        copied = rows.copy()
        rng.shuffle(copied)
        queues[name] = deque(copied)
    selected: list[dict[str, object]] = []
    names = sorted(queues)
    while len(selected) < size and any(queues.values()):
        for name in names:
            if queues[name]:
                selected.append(queues[name].popleft())
                if len(selected) == size:
                    break
    return sorted(selected, key=_row_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an unlabeled stratified qrels template")
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "data/raw/ViFinQA/questions/questions.jsonl",
    )
    parser.add_argument("--companies", type=Path, default=ROOT / "data/raw/ViFinQA/code_stock.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "annotations/qrels_template.jsonl")
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()
    if args.size <= 0:
        parser.error("--size must be positive")

    resolver = CompanyResolver.from_csv(args.companies)
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    with args.questions.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            spec = parse_query_spec(str(raw["question"]), resolver)
            scope = next((entity.scope for entity in spec.entities if entity.scope), None)
            stratum = "|".join(
                (
                    f"entities={_band(len(spec.entities))}",
                    f"years={_band(len({period.report_year for period in spec.periods}))}",
                    f"scope={scope or 'unspecified'}",
                    f"unit={spec.target_unit}",
                )
            )
            groups[stratum].append(
                {
                    "schema_version": 2,
                    "id": int(raw["id"]),
                    "question": str(raw["question"]),
                    "stratum": stratum,
                    "split_group": {
                        "entities": sorted({entity.ticker for entity in spec.entities}),
                        "years": sorted({period.report_year for period in spec.periods}),
                        "question_fingerprint": hashlib.sha256(
                            ascii_words(str(raw["question"])).encode("utf-8")
                        ).hexdigest(),
                    },
                    "answerability": "UNKNOWN",
                    "answerability_reason": "",
                    "gold_docs": [],
                    "gold_tables": [],
                    "gold_cells": [],
                    "intermediate_facts": [],
                    "cohort_members": [],
                    "operator_family": "unknown",
                    "formula_id": None,
                    "source_units": [],
                    "target_unit": spec.target_unit,
                    "program": None,
                    "answer": None,
                    "tolerance": None,
                    "rounding": None,
                    "status": "unlabeled",
                    "judgments": {
                        "annotator_a": None,
                        "annotator_b": None,
                        "adjudicator": None,
                    },
                    "adjudication_notes": "",
                    "notes": "",
                }
            )
    rows = _sample_round_robin(
        groups,
        size=min(args.size, sum(map(len, groups.values()))),
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} unlabeled rows across {len(groups)} strata to {args.output}")


if __name__ == "__main__":
    main()
