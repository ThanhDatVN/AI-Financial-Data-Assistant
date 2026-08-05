from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from openai import BadRequestError, OpenAI

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vifinqa.checkpoints.jsonl import (  # noqa: E402
    JsonlRowCheckpoint,
    write_json_atomic,
    write_jsonl_atomic,
)
from vifinqa.evidence.store import TableStore, parsed_table_to_long_frame  # noqa: E402
from vifinqa.generation.prompt import (  # noqa: E402
    CandidateSchema,
    build_program_prompt,
    numeric_cells_of,
)
from vifinqa.programs.compiler import compile_expression  # noqa: E402
from vifinqa.programs.executor import execute_expression_isolated  # noqa: E402
from vifinqa.programs.grounding import (  # noqa: E402
    cells_in_program,
    normalize_cells,
    prepare_program,
    referenced_variables,
    validate_answer_plausibility,
    validate_query_coverage,
)
from vifinqa.programs.ir import (  # noqa: E402
    BinaryExpr,
    CellExpr,
    Dimension,
    LiteralExpr,
    ScalarExpr,
)
from vifinqa.programs.serde import (  # noqa: E402
    PROGRAM_GRAMMAR_SCHEMA,
    PROGRAM_JSON_SCHEMA,
    expression_from_dict,
)

SEED = 20260802


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _append_jsonl(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _as_int(value: object, *, field: str) -> int:
    if not isinstance(value, int | str):
        raise TypeError(f"{field} must be an integer or string")
    return int(value)


def _as_str_list(value: object, *, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{field} must be a list of strings")
    return value


def _as_int_list(value: object, *, field: str) -> list[int]:
    if not isinstance(value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise TypeError(f"{field} must be a list of integers")
    return value


def _candidate_limit(row: dict[str, object], *, minimum: int) -> int:
    spec = row.get("query_spec")
    if not isinstance(spec, dict):
        raise ValueError("query_spec must be an object")
    tickers = _as_str_list(spec.get("tickers"), field="query_spec.tickers")
    years = _as_int_list(spec.get("years"), field="query_spec.years")
    return max(minimum, max(1, len(tickers)) * max(1, len(years)))


def _select_rows(
    rows: list[dict[str, object]],
    *,
    question_ids: list[int] | None,
    limit: int | None,
    shard_count: int = 1,
    shard_index: int = 0,
) -> list[dict[str, object]]:
    row_ids = [_as_int(row["id"], field="id") for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("Retrieval rows must have unique IDs")
    selected = rows
    if question_ids is not None:
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("--id values must be unique")
        rows_by_id = dict(zip(row_ids, rows, strict=True))
        missing = [question_id for question_id in question_ids if question_id not in rows_by_id]
        if missing:
            raise ValueError(f"Unknown question IDs: {missing}")
        selected = [rows_by_id[question_id] for question_id in question_ids]
    if limit is not None:
        selected = selected[:limit]
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    return [row for position, row in enumerate(selected) if position % shard_count == shard_index]


def _first_numeric_cell(frame: pd.DataFrame) -> tuple[int, int] | None:
    # A table parsed to no cells at all yields a frame with no columns, so ask before reading.
    if not {"row_index", "column_index", "numeric_value"} <= set(frame.columns):
        return None
    populated = frame.loc[frame["numeric_value"].notna(), ["row_index", "column_index"]]
    if populated.empty:
        return None
    ordered = populated.sort_values(["row_index", "column_index"]).iloc[0]
    return int(ordered["row_index"]), int(ordered["column_index"])


def _fallback_program(
    *,
    frames: dict[str, pd.DataFrame],
    target_divisor: float,
) -> tuple[str, list[str], float] | None:
    """Ground a best-effort answer on the highest-ranked evidence.

    A question the model never solved still has to appear in the submission: the organiser
    discards a file with any question missing, so an unanswered question does not cost its
    own points, it costs every point in the run. This keeps the entry executable and keeps
    its retrieval citation, which is scored separately from the number.
    """
    for variable, frame in frames.items():
        coordinate = _first_numeric_cell(frame)
        if coordinate is None:
            continue
        row_index, column_index = coordinate
        expression = normalize_cells(CellExpr(variable, row_index, column_index), {variable: frame})
        prepared: ScalarExpr = expression
        if target_divisor != 1.0:
            prepared = BinaryExpr("/", expression, LiteralExpr(target_divisor))
        query = compile_expression(prepared)
        try:
            answer = execute_expression_isolated(query, {variable: frame}, timeout_seconds=10.0)
        except Exception:  # noqa: BLE001 - a fallback must not raise
            continue
        return query, [variable], answer
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(
    *,
    retrieval: Path,
    manifest: Path,
    model: str,
    model_revision: str | None,
    candidate_tables: int,
    max_tokens: int,
    execution_timeout: float,
    request_timeout: float,
    memory_limit_mb: int | None,
    thinking_mode: str,
    max_attempts: int,
    project_revision: str | None = None,
    shard_count: int = 1,
    shard_index: int = 0,
) -> dict[str, object]:
    schema_bytes = json.dumps(PROGRAM_JSON_SCHEMA, sort_keys=True).encode()
    return {
        "retrieval_sha256": _sha256(retrieval),
        "manifest_sha256": _sha256(manifest),
        "program_schema_sha256": hashlib.sha256(schema_bytes).hexdigest(),
        "model": model,
        "model_revision": model_revision,
        "candidate_tables": candidate_tables,
        "max_tokens": max_tokens,
        "execution_timeout": execution_timeout,
        "request_timeout": request_timeout,
        "memory_limit_mb": memory_limit_mb,
        "thinking_mode": thinking_mode,
        "max_attempts": max_attempts,
        "project_revision": project_revision,
        "shard_count": shard_count,
        "shard_index": shard_index,
        "seed": SEED,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and execute grounded Pandas expressions")
    parser.add_argument("--retrieval", type=Path, default=ROOT / "outputs/retrieval.jsonl")
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data/processed/table_manifest.parquet"
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/ViFinQA")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/generation")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key-env", default="VLLM_API_KEY")
    parser.add_argument("--model", default="Qwen/Qwen3-8B-AWQ")
    parser.add_argument("--model-revision")
    parser.add_argument(
        "--thinking-mode",
        choices=("disabled", "auto"),
        default="disabled",
        help="Disable Qwen3 thinking for deterministic schema-constrained generation",
    )
    parser.add_argument(
        "--final-run",
        action="store_true",
        help="Refuse generation unless the served model revision is pinned",
    )
    parser.add_argument(
        "--candidate-tables",
        type=int,
        default=10,
        help="Minimum schemas per question; route coverage can increase this value",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        # A cohort program over the widest route fan-out spends about 1,300 tokens on its
        # coordinates alone, so a 1,024 budget truncates exactly the hardest questions.
        default=2048,
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Retry failed generation/grounding/execution with validator feedback",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        # A nine-company cohort program is the longest thing the model ever has to write, and
        # on a T4 it decodes at single-digit tokens per second. Three minutes cut those
        # questions off mid-program every time; the shorter cell form plus this ceiling gives
        # them room without letting a genuinely stuck request hold the run for ten minutes.
        default=360.0,
        help="Seconds to wait for one generation before giving the attempt up",
    )
    parser.add_argument("--execution-timeout", type=float, default=10.0)
    parser.add_argument("--memory-limit-mb", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--project-revision")
    parser.add_argument(
        "--id",
        dest="question_ids",
        action="append",
        type=int,
        help="Generate only this question ID; repeat the flag for a fixed smoke set",
    )
    args = parser.parse_args()
    if args.final_run and (
        not args.model_revision or re.fullmatch(r"[0-9a-fA-F]{40}", args.model_revision) is None
    ):
        parser.error("--final-run requires --model-revision with a full 40-character commit SHA")
    if args.final_run and (
        not args.project_revision or re.fullmatch(r"[0-9a-fA-F]{40}", args.project_revision) is None
    ):
        parser.error("--final-run requires --project-revision with the exact Git commit SHA")
    if args.candidate_tables <= 0 or args.max_tokens <= 0 or args.execution_timeout <= 0:
        parser.error("candidate tables, max tokens, and execution timeout must be positive")
    if args.request_timeout <= 0:
        parser.error("request timeout must be positive")
    if args.memory_limit_mb is not None and args.memory_limit_mb <= 0:
        parser.error("--memory-limit-mb must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.max_attempts <= 0:
        parser.error("--max-attempts must be positive")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be in [0, --shard-count)")

    rows = _select_rows(
        _load_jsonl(args.retrieval),
        question_ids=args.question_ids,
        limit=args.limit,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )
    # Load the whole ranked list, not the head of it: tables holding no number are skipped
    # below, and the budget has to be refilled from further down.
    candidate_refs = {ref for row in rows for ref in _as_str_list(row["fused"], field="fused")}
    store = TableStore.from_parquet(args.data_root, args.manifest, candidate_refs)
    args.output.mkdir(parents=True, exist_ok=True)
    data_dir = args.output / "data"
    data_dir.mkdir(exist_ok=True)
    predictions_path = args.output / "predictions.jsonl"
    errors = args.output / "errors.jsonl"
    error_attempts = args.output / "error_attempts.jsonl"
    traces = args.output / "program_traces.jsonl"
    metadata_path = args.output / "run_metadata.json"
    fingerprint = _fingerprint(
        retrieval=args.retrieval,
        manifest=args.manifest,
        model=args.model,
        model_revision=args.model_revision,
        candidate_tables=args.candidate_tables,
        max_tokens=args.max_tokens,
        execution_timeout=args.execution_timeout,
        request_timeout=args.request_timeout,
        memory_limit_mb=args.memory_limit_mb,
        thinking_mode=args.thinking_mode,
        max_attempts=args.max_attempts,
        project_revision=args.project_revision,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing != fingerprint:
            raise ValueError(
                "Output directory belongs to a different retrieval/manifest/model/schema run"
            )
    else:
        write_json_atomic(metadata_path, fingerprint)
    completed_checkpoint = JsonlRowCheckpoint(
        args.output / "completed",
        fingerprint=fingerprint,
    )
    state_rows = completed_checkpoint.load()
    if not state_rows and predictions_path.exists() and traces.exists():
        legacy_predictions = _load_jsonl(predictions_path)
        legacy_traces = _load_jsonl(traces)
        predictions_by_id = {_as_int(row["id"], field="id"): row for row in legacy_predictions}
        traces_by_id = {_as_int(row["id"], field="id"): row for row in legacy_traces}
        if len(predictions_by_id) != len(legacy_predictions) or set(predictions_by_id) != set(
            traces_by_id
        ):
            raise ValueError("Legacy prediction/program trace checkpoints are inconsistent")
        for question_id in sorted(predictions_by_id):
            completed_checkpoint.write(
                {
                    "id": question_id,
                    "prediction": predictions_by_id[question_id],
                    "trace": traces_by_id[question_id],
                }
            )
        state_rows = completed_checkpoint.load()
    completed = {_as_int(row["id"], field="id") for row in state_rows}
    unresolved_errors = (
        {_as_int(row["id"], field="id"): row for row in _load_jsonl(errors)}
        if errors.exists()
        else {}
    )
    client = OpenAI(
        base_url=args.base_url,
        api_key=os.environ.get(args.api_key_env, "local-vllm"),
        # The client default is ten minutes. A question that stalls twice then burns twenty
        # minutes of a run that has to finish 1,012 of them, so cut losses far sooner.
        timeout=args.request_timeout,
        max_retries=0,
    )

    pending_rows = [row for row in rows if _as_int(row["id"], field="id") not in completed]
    print(
        f"generation shard {args.shard_index}: {len(completed)}/{len(rows)} "
        "rows already checkpointed"
    )
    started_at = time.monotonic()
    for position, row in enumerate(pending_rows, start=1):
        question_id = _as_int(row["id"], field="id")
        candidate_limit = args.candidate_tables
        attempt_failures: list[dict[str, object]] = []
        # Bound outside the try so the fallback can still cite evidence when the failure
        # happened after the candidates were loaded.
        refs: list[str] = []
        schemas: list[CandidateSchema] = []
        frames: dict[str, pd.DataFrame] = {}
        csv_paths: dict[str, str] = {}
        table_refs: dict[str, str] = {}
        fallback_divisor = 1.0
        try:
            candidate_limit = _candidate_limit(row, minimum=args.candidate_tables)
            for table_ref in _as_str_list(row["fused"], field="fused"):
                if len(schemas) >= candidate_limit:
                    break
                record, table = store.load(str(table_ref))
                frame = parsed_table_to_long_frame(record, table)
                numeric_cells = numeric_cells_of(frame)
                # About one retrieved table in sixteen parses to no number at all, and a
                # third of questions are offered one. Every such table is a dead end that
                # also spends prompt budget, so fill the slot from further down the ranking.
                if not numeric_cells:
                    continue
                variable = f"df{len(schemas) + 1}"
                relative = f"data/q{question_id}_{variable}.csv"
                frame.to_csv(args.output / relative, index=False, encoding="utf-8")
                schemas.append(CandidateSchema(variable, record, table, numeric_cells))
                frames[variable] = frame
                csv_paths[variable] = relative
                table_refs[variable] = record.table_ref
                refs.append(record.table_ref)
            if not schemas:
                raise ValueError("No retrieved table for this question contains a parsed number")
            spec = row["query_spec"]
            if not isinstance(spec, dict):
                raise ValueError("query_spec must be an object")
            fallback_divisor = float(spec["target_divisor"])
            required_tickers = _as_str_list(spec.get("tickers"), field="query_spec.tickers")
            required_years = _as_int_list(spec.get("years"), field="query_spec.years")
            system, user = build_program_prompt(
                str(row["question"]),
                schemas,
                target_unit=str(spec["target_unit"]),
                target_divisor=float(spec["target_divisor"]),
                required_tickers=required_tickers,
                required_years=required_years,
            )
            successful: (
                tuple[dict[str, object], list[str], Dimension, str, float, list[str]] | None
            ) = None
            for attempt in range(1, args.max_attempts + 1):
                attempt_user = user
                if attempt_failures:
                    previous = attempt_failures[-1]
                    feedback = f"{previous['error_type']}: {previous['error']}"[:800]
                    attempt_user += (
                        "\n\nThe previous candidate failed deterministic validation. "
                        f"Correct it and return a complete replacement JSON. Feedback: {feedback}"
                    )
                attempt_started = time.monotonic()
                completion_tokens: int | None = None
                try:
                    extra_body: dict[str, object] = {"top_k": 20}
                    if args.thinking_mode == "disabled":
                        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
                    response = client.chat.completions.create(
                        model=args.model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": attempt_user},
                        ],
                        temperature=0.0 if attempt == 1 else 0.7,
                        top_p=1.0 if attempt == 1 else 0.8,
                        seed=SEED + attempt - 1,
                        max_tokens=args.max_tokens,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "pandas_program",
                                "strict": True,
                                "schema": PROGRAM_GRAMMAR_SCHEMA,
                            },
                        },
                        extra_body=extra_body,
                    )
                    usage = getattr(response, "usage", None)
                    completion_tokens = getattr(usage, "completion_tokens", None)
                    content = response.choices[0].message.content
                    if not content:
                        raise ValueError("Model returned empty content")
                    program = json.loads(content)
                    if not isinstance(program, dict):
                        raise ValueError("Model output must be an object")
                    expression = expression_from_dict(program.get("program"))
                    # The tree is the authority on which evidence is read. Taking the model's
                    # separate declaration at face value only loses questions to bookkeeping.
                    selected = sorted(referenced_variables(expression))
                    unknown = [variable for variable in selected if variable not in frames]
                    if unknown:
                        raise ValueError(f"Program references unknown variables: {unknown}")
                    selected_frames = {variable: frames[variable] for variable in selected}
                    prepared, inferred_dimension = prepare_program(
                        expression,
                        selected_variables=selected,
                        frames=selected_frames,
                        target_unit=str(spec["target_unit"]),
                        target_divisor=float(spec["target_divisor"]),
                    )
                    validate_query_coverage(
                        expression,
                        frames=selected_frames,
                        required_tickers=required_tickers,
                        required_years=required_years,
                    )
                    query = compile_expression(prepared)
                    answer = execute_expression_isolated(
                        query,
                        selected_frames,
                        timeout_seconds=args.execution_timeout,
                        memory_limit_mb=args.memory_limit_mb,
                    )
                    validate_answer_plausibility(answer, expression)
                    selected_refs = [table_refs[variable] for variable in selected]
                    attempt_latency = round(time.monotonic() - attempt_started, 2)
                    successful = (
                        program,
                        selected,
                        inferred_dimension,
                        query,
                        answer,
                        selected_refs,
                    )
                    break
                except Exception as attempt_exc:  # noqa: BLE001 - bounded model retry
                    # Latency and emitted tokens are the only way to tell a question that is
                    # slow from one the model never finishes, and a timeout reports neither
                    # unless it is measured here.
                    attempt_failures.append(
                        {
                            "attempt": attempt,
                            "error_type": type(attempt_exc).__name__,
                            "error": str(attempt_exc),
                            "latency_seconds": round(time.monotonic() - attempt_started, 2),
                            "completion_tokens": completion_tokens,
                        }
                    )
                    if isinstance(attempt_exc, BadRequestError):
                        raise
                    if attempt == args.max_attempts:
                        raise
            if successful is None:
                raise RuntimeError("Generation retry loop ended without a result")
            program, selected, inferred_dimension, query, answer, selected_refs = successful
            prediction: dict[str, object] = {
                "id": question_id,
                "question": str(row["question"]),
                "answer": answer,
                "relevant_docs": list(dict.fromkeys(ref.split("|", 1)[0] for ref in selected_refs)),
                "relevant_tables": selected_refs,
                "evidence": [
                    {"variable": variable, "csv_path": csv_paths[variable]} for variable in selected
                ],
                "pandas_query": query,
            }
            trace: dict[str, object] = {
                "id": question_id,
                "selected_variables": selected,
                "generated_program": program["program"],
                "inferred_dimension": inferred_dimension,
                "compiled_pandas_query": query,
                "generation_attempts": len(attempt_failures) + 1,
                "failed_attempts": attempt_failures,
                "cells_read": len(cells_in_program(expression)),
                "latency_seconds": attempt_latency,
                "completion_tokens": completion_tokens,
            }
            completed_checkpoint.write(
                {
                    "id": question_id,
                    "prediction": prediction,
                    "trace": trace,
                }
            )
            unresolved_errors.pop(question_id, None)
        except Exception as exc:  # noqa: BLE001 - batch runner records per-question failures
            error_row: dict[str, object] = {
                "id": question_id,
                "stage": "generation_or_execution",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "candidate_refs": _as_str_list(row.get("fused", []), field="fused")[
                    :candidate_limit
                ],
                "model": args.model,
                "model_revision": args.model_revision,
                "thinking_mode": args.thinking_mode,
                "failed_attempts": attempt_failures,
                "run_fingerprint": fingerprint,
            }
            unresolved_errors[question_id] = error_row
            _append_jsonl(error_attempts, error_row)
            fallback = _fallback_program(frames=frames, target_divisor=fallback_divisor)
            if fallback is not None:
                query, selected, answer = fallback
                selected_refs = [table_refs[variable] for variable in selected]
                completed_checkpoint.write(
                    {
                        "id": question_id,
                        "prediction": {
                            "id": question_id,
                            "question": str(row["question"]),
                            "answer": answer,
                            "relevant_docs": list(
                                dict.fromkeys(ref.split("|", 1)[0] for ref in selected_refs)
                            ),
                            "relevant_tables": selected_refs,
                            "evidence": [
                                {"variable": variable, "csv_path": csv_paths[variable]}
                                for variable in selected
                            ],
                            "pandas_query": query,
                        },
                        "trace": {
                            "id": question_id,
                            "fallback": True,
                            "fallback_reason": f"{type(exc).__name__}: {exc}"[:400],
                            "selected_variables": selected,
                            "compiled_pandas_query": query,
                            "generation_attempts": len(attempt_failures),
                            "failed_attempts": attempt_failures,
                        },
                    }
                )
        if position == 1 or position % 10 == 0 or position == len(pending_rows):
            elapsed = max(time.monotonic() - started_at, 1e-9)
            remaining = len(pending_rows) - position
            eta_hours = remaining / (position / elapsed) / 3_600
            print(
                f"generation shard {args.shard_index}: {position}/{len(pending_rows)} new rows; "
                f"{position / elapsed:.3f} questions/s; ETA {eta_hours:.2f} h"
            )

    state_rows = completed_checkpoint.load()
    predictions: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    for state in state_rows:
        stored_prediction = state.get("prediction")
        stored_trace = state.get("trace")
        if not isinstance(stored_prediction, dict) or not isinstance(stored_trace, dict):
            raise TypeError("Completed state must contain prediction and trace objects")
        predictions.append(stored_prediction)
        trace_rows.append(stored_trace)
    predictions.sort(key=lambda item: _as_int(item["id"], field="id"))
    trace_rows.sort(key=lambda item: _as_int(item["id"], field="id"))
    write_jsonl_atomic(predictions_path, predictions)
    write_jsonl_atomic(traces, trace_rows)
    final_trace_ids = {_as_int(item["id"], field="id") for item in trace_rows}
    prediction_ids = {_as_int(item["id"], field="id") for item in predictions}
    write_jsonl_atomic(
        errors,
        [
            error
            for question_id, error in sorted(unresolved_errors.items())
            if question_id not in prediction_ids
        ],
    )
    if final_trace_ids != prediction_ids:
        raise ValueError("Every prediction must have exactly one program trace")
    write_json_atomic(args.output / "submission.json", predictions)
    print(f"completed={len(predictions)}/{len(rows)}; output={args.output}")


if __name__ == "__main__":
    main()
