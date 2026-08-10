from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from vifinqa.submission.package import package_submission
from vifinqa.submission.validate import validate_submission


def test_submission_is_executed_and_packaged_at_zip_root(tmp_path: Path) -> None:
    questions = tmp_path / "questions.jsonl"
    questions.write_text('{"id":1,"question":"Q?"}\n', encoding="utf-8")
    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame({"x": [2.0]}).to_csv(data / "one.csv", index=False)
    submission = tmp_path / "candidate.json"
    submission.write_text(
        json.dumps(
            [
                {
                    "id": 1,
                    "question": "Q?",
                    "answer": 2.0,
                    "relevant_docs": ["DOC"],
                    "relevant_tables": ["DOC|table_1"],
                    "evidence": [{"variable": "df1", "csv_path": "data/one.csv"}],
                    "pandas_query": "float(df1['x'].iloc[0])",
                }
            ]
        ),
        encoding="utf-8",
    )
    assert len(validate_submission(submission, questions_path=questions)) == 1
    output = package_submission(submission, tmp_path / "submission.zip", questions_path=questions)
    with ZipFile(output) as archive:
        assert set(archive.namelist()) == {"submission.json", "data/one.csv"}


def _single_question_submission(tmp_path: Path, answer: float) -> tuple[Path, Path]:
    frame = pd.DataFrame(
        {
            "row_index": [0],
            "column_index": [0],
            "base_value": [4.7732000000157094e38],
            "numeric_value": [4.7732000000157094e38],
        }
    )
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    frame.to_csv(data / "q1_df1.csv", index=False, encoding="utf-8")
    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        json.dumps({"id": 1, "question": "Bao nhieu?"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    query = (
        "float(df1.loc[(df1['row_index'] == 0) & "
        "(df1['column_index'] == 0), 'base_value'].iloc[0])"
    )
    submission = tmp_path / "submission.json"
    submission.write_text(
        json.dumps(
            [
                {
                    "id": 1,
                    "question": "Bao nhieu?",
                    "answer": answer,
                    "relevant_docs": ["DOC"],
                    "relevant_tables": ["DOC|table_1"],
                    "evidence": [{"variable": "df1", "csv_path": "data/q1_df1.csv"}],
                    "pandas_query": query,
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return submission, questions


def test_execution_check_scales_with_the_magnitude_it_is_checking(tmp_path: Path) -> None:
    """A disagreement of one part in 10^16 is the same answer.

    One question answered 4.7732000000157094e+38, whose last representable step is about
    1e22. Round-tripping its evidence through CSV moved it by one of those, and an absolute
    tolerance of 0.01 called a submission that reproduced perfectly a mismatch.
    """
    submission, questions = _single_question_submission(tmp_path, 4.773200000015709e38)
    predictions = validate_submission(submission, questions_path=questions, evidence_root=tmp_path)
    assert len(predictions) == 1

    # A percent off is not a rounding artefact and still has to fail.
    submission, questions = _single_question_submission(tmp_path, 4.82e38)
    with pytest.raises(ValueError, match="Execution mismatch"):
        validate_submission(submission, questions_path=questions, evidence_root=tmp_path)
