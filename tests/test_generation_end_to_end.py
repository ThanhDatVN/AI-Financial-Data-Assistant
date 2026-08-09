"""Drive the whole generation path against a stub model, with no GPU.

The disaster this guards against is not a wrong answer. The organiser discards a submission
with any question missing, so a run where the model solves nothing has to still produce a
complete, executable, packageable file. That path only exists end to end, so it is worth
exercising end to end.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pandas as pd
import pytest

from vifinqa.evidence.store import TableStore, parsed_table_to_long_frame
from vifinqa.submission.package import package_submission
from vifinqa.submission.validate import validate_submission

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/raw/ViFinQA"
MANIFEST = ROOT / "data/processed/table_manifest.parquet"
RETRIEVAL = ROOT / "outputs/retrieval.jsonl"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not DATA_ROOT.exists() or not MANIFEST.exists() or not RETRIEVAL.exists(),
        reason="local ViFinQA corpus and frozen retrieval are optional",
    ),
]


class _UnsolvableModel(BaseHTTPRequestHandler):
    """Answer every request with a well-formed program that grounding must reject."""

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        program = {
            "selected_variables": ["df1"],
            "program": {
                "kind": "cell",
                "variable": "df99",
                "row_index": 0,
                "column_index": 0,
            },
        }
        body = json.dumps(
            {
                "id": "stub",
                "object": "chat.completion",
                "model": "stub/model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": json.dumps(program)},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_: object) -> None:
        return


class _SolvingModel(BaseHTTPRequestHandler):
    """Answer with the coordinate the runner's own evidence says is populated."""

    coordinate: tuple[int, int] = (0, 0)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        row_index, column_index = _SolvingModel.coordinate
        program = {
            "selected_variables": ["df1"],
            "program": {
                "kind": "cell",
                "variable": "df1",
                "row_index": row_index,
                "column_index": column_index,
            },
        }
        body = json.dumps(
            {
                "id": "stub",
                "object": "chat.completion",
                "model": "stub/model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": json.dumps(program)},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_: object) -> None:
        return


def _first_candidate_with_numbers(row: dict[str, object]) -> tuple[str, tuple[int, int]]:
    """Reproduce the runner's choice of df1 and the first populated cell inside it."""
    refs = [str(ref) for ref in row["fused"]]  # type: ignore[union-attr]
    store = TableStore.from_parquet(DATA_ROOT, MANIFEST, set(refs))
    for table_ref in refs:
        record, table = store.load(table_ref)
        frame = parsed_table_to_long_frame(record, table)
        if "numeric_value" not in frame.columns:
            continue
        populated = frame.loc[frame["numeric_value"].notna(), ["row_index", "column_index"]]
        if populated.empty:
            continue
        first = populated.sort_values(["row_index", "column_index"]).iloc[0]
        return table_ref, (int(first["row_index"]), int(first["column_index"]))
    raise AssertionError("no candidate table holds a number")


def test_a_solved_question_becomes_a_reproducible_prediction(tmp_path: Path) -> None:
    row = next(
        json.loads(line) for line in RETRIEVAL.read_text(encoding="utf-8").splitlines() if line
    )
    table_ref, _SolvingModel.coordinate = _first_candidate_with_numbers(row)
    retrieval = tmp_path / "retrieval.jsonl"
    retrieval.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        json.dumps({"id": row["id"], "question": row["question"]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    server = HTTPServer(("127.0.0.1", 0), _SolvingModel)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output = tmp_path / "generation"
    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/50_generate_programs.py"),
                "--retrieval",
                str(retrieval),
                "--manifest",
                str(MANIFEST),
                "--data-root",
                str(DATA_ROOT),
                "--output",
                str(output),
                "--base-url",
                f"http://127.0.0.1:{server.server_port}/v1",
                "--model",
                "stub/model",
                "--max-attempts",
                "1",
                "--candidate-tables",
                "3",
                "--request-timeout",
                "60",
            ],
            capture_output=True,
            text=True,
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
    assert completed.returncode == 0, completed.stderr[-4000:]

    traces = [
        json.loads(line)
        for line in (output / "program_traces.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(traces) == 1
    assert not traces[0].get("fallback"), "a grounded program must not be recorded as a fallback"
    assert traces[0]["cells_read"] == 1

    predictions = validate_submission(
        output / "submission.json",
        questions_path=questions,
        evidence_root=output,
        execute=True,
    )
    assert len(predictions) == 1
    assert predictions[0].relevant_tables == [table_ref]
    assert predictions[0].answer == predictions[0].answer  # a real number, never NaN


def test_a_run_that_solves_nothing_still_packages_every_question(tmp_path: Path) -> None:
    rows = [
        json.loads(line)
        for line in RETRIEVAL.read_text(encoding="utf-8").splitlines()[:3]
        if line.strip()
    ]
    retrieval = tmp_path / "retrieval.jsonl"
    retrieval.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        "".join(
            json.dumps({"id": row["id"], "question": row["question"]}, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )

    server = HTTPServer(("127.0.0.1", 0), _UnsolvableModel)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output = tmp_path / "generation"
    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/50_generate_programs.py"),
                "--retrieval",
                str(retrieval),
                "--manifest",
                str(MANIFEST),
                "--data-root",
                str(DATA_ROOT),
                "--output",
                str(output),
                "--base-url",
                f"http://127.0.0.1:{server.server_port}/v1",
                "--model",
                "stub/model",
                "--max-attempts",
                "1",
                "--candidate-tables",
                "3",
                "--request-timeout",
                "60",
            ],
            capture_output=True,
            text=True,
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
    assert completed.returncode == 0, completed.stderr[-4000:]

    expected_ids = {int(row["id"]) for row in rows}
    traces = [
        json.loads(line)
        for line in (output / "program_traces.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert {int(trace["id"]) for trace in traces} == expected_ids
    assert all(trace["fallback"] for trace in traces), "every question should have fallen back"

    predictions = validate_submission(
        output / "submission.json",
        questions_path=questions,
        evidence_root=output,
        execute=True,
    )
    assert {prediction.id for prediction in predictions} == expected_ids
    for prediction in predictions:
        assert prediction.relevant_tables, "a fallback still has to cite its evidence"
        assert all(item.csv_path.startswith("data/") for item in prediction.evidence)

    archive = package_submission(
        output / "submission.json",
        tmp_path / "submission.zip",
        questions_path=questions,
        evidence_root=output,
    )
    assert archive.is_file() and archive.stat().st_size > 0


def test_evidence_csvs_rebuild_from_the_rows_that_cite_them(tmp_path: Path) -> None:
    # Carrying the evidence between sessions costs gigabytes and downloading it defeated the
    # browser; the tables are a deterministic function of the corpus, so move only the rows.
    table_ref = "VJC_financial_statements_2018_separate|table_50"
    shard = tmp_path / "shard_0"
    (shard / "rows").mkdir(parents=True)
    (shard / "rows" / "00000001.json").write_text(
        json.dumps(
            {
                "id": 1,
                "prediction": {
                    "id": 1,
                    "relevant_tables": [table_ref],
                    "evidence": [{"variable": "df1", "csv_path": "data/q1_df1.csv"}],
                },
                "trace": {"id": 1},
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/52_restore_evidence_csv.py"),
            str(shard),
            "--manifest",
            str(MANIFEST),
            "--data-root",
            str(DATA_ROOT),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]

    restored = shard / "data/q1_df1.csv"
    assert restored.is_file()
    frame = pd.read_csv(restored)
    cell = frame.loc[(frame["row_index"] == 0) & (frame["column_index"] == 1)]
    assert float(cell["base_value"].iloc[0]) == 208253201298.0

    # A second pass has nothing left to do and must not rewrite what is already there.
    again = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/52_restore_evidence_csv.py"),
            str(shard),
            "--manifest",
            str(MANIFEST),
            "--data-root",
            str(DATA_ROOT),
        ],
        capture_output=True,
        text=True,
    )
    assert again.returncode == 0, again.stderr[-4000:]
    assert "missing 0" in again.stdout
