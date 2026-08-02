from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

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
