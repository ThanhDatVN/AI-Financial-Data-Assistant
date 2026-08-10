from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from vifinqa.programs.executor import execute_expression_isolated
from vifinqa.submission.schema import Prediction


def _load_questions(path: Path) -> dict[int, str]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {int(row["id"]): str(row["question"]) for row in rows}


def _safe_csv_path(root: Path, relative: str) -> Path:
    path = (root / Path(relative)).resolve()
    data_root = (root / "data").resolve()
    if path.parent != data_root and data_root not in path.parents:
        raise ValueError(f"Evidence path escapes data/: {relative}")
    return path


def validate_submission(
    submission_path: Path,
    *,
    questions_path: Path,
    evidence_root: Path | None = None,
    execute: bool = True,
    abs_tolerance: float = 0.01,
    execution_timeout: float = 10.0,
    allow_partial_docs: bool = False,
) -> list[Prediction]:
    raw = json.loads(submission_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Submission root must be a JSON list")
    context = {"allow_partial_docs": allow_partial_docs}
    predictions = [Prediction.model_validate(item, context=context) for item in raw]
    expected = _load_questions(questions_path)
    by_id = {prediction.id: prediction for prediction in predictions}
    if len(by_id) != len(predictions):
        raise ValueError("Duplicate question IDs in submission")
    if set(by_id) != set(expected):
        missing = sorted(set(expected) - set(by_id))
        extra = sorted(set(by_id) - set(expected))
        raise ValueError(f"Question ID mismatch; missing={missing[:20]}, extra={extra[:20]}")

    root = evidence_root or submission_path.parent
    for question_id, prediction in by_id.items():
        if prediction.question != expected[question_id]:
            raise ValueError(f"Question text mismatch for id={question_id}")
        frames: dict[str, pd.DataFrame] = {}
        for evidence in prediction.evidence:
            csv_path = _safe_csv_path(root, evidence.csv_path)
            if not csv_path.is_file():
                raise ValueError(f"Missing evidence CSV for id={question_id}: {evidence.csv_path}")
            frames[evidence.variable] = pd.read_csv(csv_path)
        if execute:
            actual = execute_expression_isolated(
                prediction.pandas_query,
                frames,
                timeout_seconds=execution_timeout,
            )
            # An absolute tolerance is the right test near zero and meaningless far from it:
            # one unit in the last place of 4.8e38 is about 1e22, so a value that survived a
            # round trip through CSV and back failed a check it reproduced perfectly. Accept
            # either an absolute or a relative agreement; the relative one is still nine
            # significant digits, far tighter than any answer tolerance the task implies.
            if not math.isclose(actual, prediction.answer, rel_tol=1e-9, abs_tol=abs_tolerance):
                raise ValueError(
                    f"Execution mismatch for id={question_id}: "
                    f"declared={prediction.answer}, actual={actual}"
                )
    return predictions
