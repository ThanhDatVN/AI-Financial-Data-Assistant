"""Compare two matched generation roots without pretending to have hidden gold labels.

The variants must contain exactly the same question IDs.  The report quantifies answer,
program, evidence, fallback, latency and token differences and records hashes for provenance.
It is descriptive only: a lower fallback rate is useful evidence, but only a matched public
score or trusted labels can establish which answers are correct.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise TypeError(f"{path} must contain JSON objects")
    return rows


def _indexed_submission(root: Path) -> dict[int, dict[str, Any]]:
    path = root / "submission.json"
    rows = _load_json(path)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise TypeError(f"{path} must be a JSON list of objects")
    indexed = {int(row["id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate IDs in {path}")
    return indexed


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _variant_summary(root: Path, rows: dict[int, dict[str, Any]]) -> dict[str, Any]:
    traces = _load_jsonl(root / "program_traces.jsonl")
    errors = _load_jsonl(root / "errors.jsonl")
    latencies = [
        float(trace["latency_seconds"])
        for trace in traces
        if isinstance(trace.get("latency_seconds"), int | float)
    ]
    completion_tokens = [
        int(trace["completion_tokens"])
        for trace in traces
        if isinstance(trace.get("completion_tokens"), int | float)
    ]
    metadata_path = root / "run_metadata.json"
    metadata = _load_json(metadata_path) if metadata_path.is_file() else {}
    return {
        "root": str(root),
        "rows": len(rows),
        "submission_sha256": _sha256(root / "submission.json"),
        "run_metadata_sha256": _sha256(metadata_path) if metadata_path.is_file() else None,
        "table_unit_source": metadata.get("table_unit_source")
        if isinstance(metadata, dict)
        else None,
        "traces": len(traces),
        "errors": len(errors),
        "fallbacks": sum(bool(trace.get("fallback")) for trace in traces),
        "median_latency_seconds": _median(latencies),
        "median_completion_tokens": _median([float(value) for value in completion_tokens]),
    }


def compare_roots(left_root: Path, right_root: Path) -> dict[str, Any]:
    left = _indexed_submission(left_root)
    right = _indexed_submission(right_root)
    left_ids = set(left)
    right_ids = set(right)
    if left_ids != right_ids:
        missing_right = sorted(left_ids - right_ids)
        missing_left = sorted(right_ids - left_ids)
        raise ValueError(
            f"Variant ID mismatch; missing_right={missing_right[:20]}, "
            f"missing_left={missing_left[:20]}"
        )

    changed_answers: list[int] = []
    changed_queries: list[int] = []
    changed_evidence: list[int] = []
    changed_tables: list[int] = []
    changed_docs: list[int] = []
    for question_id in sorted(left_ids):
        left_row = left[question_id]
        right_row = right[question_id]
        left_answer = float(left_row["answer"])
        right_answer = float(right_row["answer"])
        if not math.isclose(left_answer, right_answer, rel_tol=1e-12, abs_tol=1e-9):
            changed_answers.append(question_id)
        if left_row.get("pandas_query") != right_row.get("pandas_query"):
            changed_queries.append(question_id)
        if left_row.get("evidence") != right_row.get("evidence"):
            changed_evidence.append(question_id)
        if left_row.get("relevant_tables") != right_row.get("relevant_tables"):
            changed_tables.append(question_id)
        if left_row.get("relevant_docs") != right_row.get("relevant_docs"):
            changed_docs.append(question_id)

    changed_any = sorted(
        set(changed_answers)
        | set(changed_queries)
        | set(changed_evidence)
        | set(changed_tables)
        | set(changed_docs)
    )
    return {
        "descriptive_only": True,
        "warning": "No hidden gold labels were used; this report cannot select the winner.",
        "question_ids_sha256": hashlib.sha256(
            ("\n".join(map(str, sorted(left_ids))) + "\n").encode()
        ).hexdigest(),
        "questions": len(left_ids),
        "left": _variant_summary(left_root, left),
        "right": _variant_summary(right_root, right),
        "differences": {
            "answers": len(changed_answers),
            "pandas_queries": len(changed_queries),
            "evidence": len(changed_evidence),
            "relevant_tables": len(changed_tables),
            "relevant_docs": len(changed_docs),
            "any": len(changed_any),
            "answer_ids": changed_answers,
            "query_ids": changed_queries,
            "any_ids": changed_any,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = compare_roots(args.left, args.right)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    differences = report["differences"]
    print(
        f"questions={report['questions']} changed_answers={differences['answers']} "
        f"changed_queries={differences['pandas_queries']} changed_any={differences['any']}"
    )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
