"""Enumerate, without a GPU, every question the generator cannot possibly answer.

Only the model needs an accelerator. Routing, evidence, grounding, dimensions and the
program grammar are all deterministic, so the constraints a program must satisfy can be
checked against the candidates a question actually receives. Whatever fails here fails for
every model on every attempt, and no prompt or timeout changes it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vifinqa.checkpoints.jsonl import write_json_atomic  # noqa: E402
from vifinqa.programs.grounding import TARGET_DIMENSIONS  # noqa: E402
from vifinqa.programs.serde import GRAMMAR_MAX_ITEMS  # noqa: E402

# One cell node costs about 65 characters, and roughly three characters make a token.
TOKENS_PER_CELL = 22

# Question shapes that need the select node: a subset chosen by a predicate, or a quantile.
# These are answerable, but only through that one operator, so they are worth counting.
_NEEDS_SELECTION = re.compile(
    r"trung v[ịi]|ph[âa]n v[ịi]|t[ứu] ph[âa]n"
    r"|trong c[ảa] (?:ba|hai|b[ốo]n|\d)\s*n[ăa]m"
    r"|trung b[ìi]nh c[ủu]a (?:c[ảa]|nh[óo]m)"
)


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).lower()


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _candidate_limit(tickers: list[str], years: list[int], minimum: int) -> int:
    return max(minimum, max(1, len(tickers)) * max(1, len(years)))


_UNIT_DIMENSION = {
    "VND": "VND",
    "THOUSAND_VND": "VND",
    "MILLION_VND": "VND",
    "BILLION_VND": "VND",
    "USD": "USD",
    "MILLION_USD": "USD",
    "PERCENT": "PERCENT",
    "SHARES": "SHARES",
}


def _dimension_unreachable(target: str, units: list[str]) -> bool:
    """Whether no program over these candidates can produce the target dimension.

    Currency never converts without rate evidence, so a VND answer needs a VND cell. A
    percentage or ratio can also be built by dividing two cells of one dimension. Counts and
    years come from operators and literals, so candidates never block them, and a table whose
    own unit is unknown may still hold cells that carry one.
    """
    dimension = TARGET_DIMENSIONS.get(target)
    if dimension in {None, "COUNT", "YEAR"}:
        return False
    resolved = [_UNIT_DIMENSION.get(unit, "UNKNOWN") for unit in units]
    if "UNKNOWN" in resolved:
        return False
    if dimension in {"PERCENT", "RATIO"}:
        if "PERCENT" in resolved:
            return False
        return not any(resolved.count(other) >= 2 for other in set(resolved))
    return dimension not in resolved


def _audit_row(
    row: dict[str, object],
    metadata: pd.DataFrame,
    *,
    minimum_candidates: int,
    max_tokens: int,
) -> dict[str, object]:
    spec = row["query_spec"]
    assert isinstance(spec, dict)
    tickers = [str(item) for item in spec["tickers"]]
    years = [int(item) for item in spec["years"]]
    target_unit = str(spec["target_unit"])
    fused = row["fused"]
    assert isinstance(fused, list)
    refs = [str(ref) for ref in fused][: _candidate_limit(tickers, years, minimum_candidates)]
    candidates = metadata.reindex(refs).dropna(subset=["ticker"])

    available_tickers = set(candidates["ticker"].astype(str))
    available_years = set(candidates["report_year"].astype(int))
    # A prior-year column lets the next report answer for the year before it.
    covered_years = available_years | {year - 1 for year in available_years}

    blockers: list[str] = []
    if target_unit not in TARGET_DIMENSIONS:
        blockers.append("unsupported_target_unit")
    missing_tickers = sorted(set(tickers) - available_tickers)
    if missing_tickers:
        blockers.append("candidates_miss_required_ticker")
    missing_years = sorted(set(years) - covered_years)
    if missing_years:
        blockers.append("candidates_miss_required_year")
    candidate_units = [str(unit) for unit in candidates["unit"]]
    if _dimension_unreachable(target_unit, candidate_units):
        blockers.append("no_candidate_carries_the_target_dimension")

    cells_needed = max(1, len(tickers)) * max(1, len(years))
    if cells_needed > GRAMMAR_MAX_ITEMS:
        blockers.append("cohort_wider_than_grammar")
    if cells_needed * TOKENS_PER_CELL > max_tokens:
        blockers.append("program_longer_than_token_budget")

    return {
        "id": int(str(row["id"])),
        "needs_selection": bool(_NEEDS_SELECTION.search(_normalize(str(row["question"])))),
        "target_unit": target_unit,
        "required_tickers": tickers,
        "required_years": years,
        "candidates": len(refs),
        "candidates_resolved": int(len(candidates)),
        "missing_tickers": missing_tickers,
        "missing_years": missing_years,
        "cells_needed": cells_needed,
        "estimated_tokens": cells_needed * TOKENS_PER_CELL,
        "blockers": blockers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, default=ROOT / "outputs/retrieval.jsonl")
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data/processed/table_manifest.parquet"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/generation_feasibility.json")
    parser.add_argument("--candidate-tables", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()

    rows = _load_jsonl(args.retrieval)
    metadata = (
        pd.read_parquet(args.manifest, columns=["table_ref", "ticker", "report_year", "unit"])
        .drop_duplicates("table_ref")
        .set_index("table_ref")
    )

    audited = [
        _audit_row(
            row,
            metadata,
            minimum_candidates=args.candidate_tables,
            max_tokens=args.max_tokens,
        )
        for row in rows
    ]
    blocked = [record for record in audited if record["blockers"]]
    causes: Counter[str] = Counter()
    for record in blocked:
        blockers = record["blockers"]
        assert isinstance(blockers, list)
        causes.update(str(blocker) for blocker in blockers)

    report = {
        "format_version": 1,
        "retrieval": str(args.retrieval),
        "questions": len(audited),
        "blocked": len(blocked),
        "reachable": len(audited) - len(blocked),
        "causes": dict(causes.most_common()),
        "blocked_ids_by_cause": {
            cause: [
                record["id"]
                for record in blocked
                if cause in list(record["blockers"])  # type: ignore[call-overload]
            ][:40]
            for cause in causes
        },
        "questions_detail": audited,
    }
    write_json_atomic(args.output, report)

    needing = sum(1 for record in audited if record["needs_selection"])
    report["needs_selection"] = needing
    print(f"questions           {report['questions']}")
    print(f"need select node    {needing}")
    print(f"reachable           {report['reachable']}")
    print(f"blocked             {report['blocked']}")
    for cause, count in causes.most_common():
        print(f"  {count:4d}  {cause}")
    print(f"report              {args.output}")


if __name__ == "__main__":
    main()
