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
from vifinqa.parsing.normalize import ascii_words  # noqa: E402
from vifinqa.parsing.units import TABLE_UNIT_INFERENCE_VERSION  # noqa: E402
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
from vifinqa.submission.semantics import (  # noqa: E402
    SEMANTIC_CONVENTION_VERSION,
    normalize_absolute_difference,
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


_CONTEXT_LIMIT_RE = re.compile(r"maximum context length is (\d+) tokens")
_INPUT_TOKENS_RE = re.compile(r"value=(\d+)")


class _Truncated(ValueError):
    """The model ran out of budget mid-program.

    Worth its own type because the remedy is the opposite of every other failure's: the program
    is not wrong and must not be criticised back to the model, it is simply unfinished.
    """


def _context_numbers(message: str) -> tuple[int, int] | None:
    """Read the context limit and the prompt's length off a context-length refusal."""
    limit = _CONTEXT_LIMIT_RE.search(message)
    used = _INPUT_TOKENS_RE.search(message)
    if not limit or not used:
        return None
    return int(limit.group(1)), int(used.group(1))


def _room_for_output(context_limit: int, prompt_tokens: int, *, margin: int) -> int:
    """How many output tokens this prompt can still afford, with room to spare."""
    return context_limit - prompt_tokens - margin


class _TokenBudget:
    """How many output tokens this one question gets, revised as the server reveals the truth.

    A fixed budget is wrong in both directions at once. Question 213 carries a 12.4k-token prompt,
    so a 4,096 budget puts the request past the 16,384 context and is refused outright. Question
    442's prompt is short enough to have afforded far more, and all three of its attempts stopped
    mid-string at exactly 4,096 tokens and failed to parse.

    Guessing from the refusal alone did not converge: the refusal reports the prompt as "at least"
    so many tokens, and question 213's corrected request came back one token over a second time.
    So prefer the figure that is not a lower bound -- `usage.prompt_tokens` off any response the
    server did return -- and widen the safety margin on each successive refusal.
    """

    def __init__(self, ceiling: int, context_limit: int, *, floor: int = 256) -> None:
        self.ceiling = ceiling
        self.context_limit = context_limit
        self.floor = floor
        self.margin = 64
        self.current = ceiling

    def observe(self, prompt_tokens: int | None) -> None:
        """Take the exact prompt length from a response the server actually produced.

        Only ever lowers. Raising here would undo `widen` on the very next attempt -- the
        measurement would clamp the budget back to the ceiling that truncated the program in the
        first place, and the retry would stop at the same place for the same reason.
        """
        if prompt_tokens is None:
            return
        room = _room_for_output(self.context_limit, prompt_tokens, margin=self.margin)
        if room < self.current:
            self.current = max(self.floor, room)

    def shrink(self, message: str) -> bool:
        """Fit the budget inside a context the server just refused. False when it cannot."""
        numbers = _context_numbers(message)
        if numbers is None:
            return False
        self.context_limit, prompt_tokens = numbers
        # Each refusal doubles the cushion. One token short only buys another refusal, and a
        # question that spends its attempts re-measuring its own prompt answers nothing.
        self.margin *= 2
        room = _room_for_output(self.context_limit, prompt_tokens, margin=self.margin)
        if room < self.floor or room >= self.current:
            return False
        self.current = room
        return True

    def widen(self, prompt_tokens: int | None) -> bool:
        """Give a truncated answer the rest of the context. False when none is left.

        A program cut off mid-string is not a wrong program, it is an unfinished one, and
        retrying it at the same budget produces the same cut in the same place three times over.
        """
        if prompt_tokens is None:
            return False
        room = _room_for_output(self.context_limit, prompt_tokens, margin=self.margin)
        if room <= self.current:
            return False
        self.current = room
        return True


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


_STOPWORDS = frozenset(
    """la bao nhieu cua nam cong ty co phan va trong theo duoc doanh nghiep tai gia tri
    tong so cac cho tu den moi""".split()
)


def _content_words(text: object) -> set[str]:
    return {word for word in ascii_words(str(text)).split() if len(word) > 2} - _STOPWORDS


def _best_matching_cell(
    frames: dict[str, pd.DataFrame],
    *,
    question: str,
    years: list[int],
) -> tuple[str, int, int] | None:
    """Pick the cell whose labels answer the question most nearly.

    The first populated cell of the first table is an arbitrary number wearing the shape of
    an answer. Matching the question's own words against row labels, and its years against
    column headers, at least aims at the figure it asked for. On both questions whose
    correct value is known this selects the right table and the right cell, even though one
    of them sits eighth in the ranking.
    """
    asked = _content_words(question)
    wanted = {str(year) for year in years}
    best: tuple[int, str, int, int] | None = None
    for variable, frame in frames.items():
        if not {"row_index", "column_index", "numeric_value"} <= set(frame.columns):
            continue
        populated = frame.loc[frame["numeric_value"].notna()]
        for row in populated.itertuples(index=False):
            score = 2 * len(asked & _content_words(getattr(row, "row_label", "")))
            if any(year in str(getattr(row, "column_label", "")) for year in wanted):
                score += 3
            coordinate = (score, variable, int(row.row_index), int(row.column_index))
            # Ties fall to the earliest variable and coordinate, so the choice is stable.
            if best is None or coordinate[0] > best[0]:
                best = coordinate
    if best is None:
        return None
    _, variable, row_index, column_index = best
    return variable, row_index, column_index


def _fallback_program(
    *,
    frames: dict[str, pd.DataFrame],
    target_divisor: float,
    question: str = "",
    years: list[int] | None = None,
) -> tuple[str, list[str], float] | None:
    """Ground a best-effort answer on the evidence that best matches the question.

    A question the model never solved still has to appear in the submission: the organiser
    discards a file with any question missing, so an unanswered question does not cost its
    own points, it costs every point in the run. This keeps the entry executable and keeps
    its retrieval citation, which is scored separately from the number.
    """
    selection = _best_matching_cell(frames, question=question, years=years or [])
    if selection is None:
        return None
    variable, row_index, column_index = selection
    frame = frames[variable]
    expression = normalize_cells(CellExpr(variable, row_index, column_index), {variable: frame})
    prepared: ScalarExpr = expression
    if target_divisor != 1.0:
        prepared = BinaryExpr("/", expression, LiteralExpr(target_divisor))
    query = compile_expression(prepared)
    try:
        answer = execute_expression_isolated(query, {variable: frame}, timeout_seconds=10.0)
    except Exception:  # noqa: BLE001 - a fallback must not raise
        return None
    return query, [variable], answer


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
    context_limit: int,
    execution_timeout: float,
    request_timeout: float,
    memory_limit_mb: int | None,
    thinking_mode: str,
    max_attempts: int,
    selected_question_ids: list[int] | None = None,
    table_unit_source: str = "latest",
    row_hierarchy: bool = False,
    project_revision: str | None = None,
    shard_count: int = 1,
    shard_index: int = 0,
) -> dict[str, object]:
    schema_bytes = json.dumps(PROGRAM_JSON_SCHEMA, sort_keys=True).encode()
    question_ids_bytes = json.dumps(selected_question_ids or [], separators=(",", ":")).encode()
    return {
        "retrieval_sha256": _sha256(retrieval),
        "manifest_sha256": _sha256(manifest),
        "program_schema_sha256": hashlib.sha256(schema_bytes).hexdigest(),
        "semantic_convention_version": SEMANTIC_CONVENTION_VERSION,
        "table_unit_inference_version": TABLE_UNIT_INFERENCE_VERSION,
        "question_count": len(selected_question_ids or []),
        "question_ids_sha256": hashlib.sha256(question_ids_bytes).hexdigest(),
        "table_unit_source": table_unit_source,
        "model": model,
        "model_revision": model_revision,
        "candidate_tables": candidate_tables,
        "max_tokens": max_tokens,
        "context_limit": context_limit,
        "execution_timeout": execution_timeout,
        "request_timeout": request_timeout,
        "memory_limit_mb": memory_limit_mb,
        "thinking_mode": thinking_mode,
        "max_attempts": max_attempts,
        "row_hierarchy": row_hierarchy,
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
    parser.add_argument(
        "--model",
        # The notebooks always pass this explicitly, so the default only matters for a bare
        # CLI call. It tracks whichever model the pipeline currently runs, because a default
        # naming a retired model is a trap for anyone reproducing a run by hand.
        default="Qwen/Qwen3-14B-AWQ",
    )
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
        "--row-hierarchy",
        action="store_true",
        help="Show conservative accounting parent paths in row labels (experimental ablation)",
    )
    parser.add_argument(
        "--table-unit-source",
        choices=("latest", "manifest"),
        default="latest",
        help="Use latest parser units or frozen manifest units for a controlled ablation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        # A cohort program over the widest route fan-out spends about 1,300 tokens on its
        # coordinates alone, so a 1,024 budget truncates exactly the hardest questions.
        #
        # 2,048 truncated them too. Questions 399 and 442 stopped at exactly 2,048 completion
        # tokens on all three attempts and failed to parse mid-JSON every time -- an unterminated
        # string, then a missing property name, then a missing value. With the Marlin kernel
        # decoding at 22 tokens per second, 4,096 takes 186 seconds against a 360-second request
        # timeout, and the longest prompt measured over thirty questions is 9,337 tokens, so
        # prompt plus budget stays under the 16,384 context with room to spare.
        # 4,096 stopped truncating question 399 but not 442, whose prompt is short enough to
        # afford more. At 22 tokens per second 6,144 costs 278 seconds inside a 360-second
        # timeout, while 8,192 would cost 371 and time out on its own; the context correction
        # above trims this back for questions whose prompts leave less room.
        default=6144,
    )
    parser.add_argument(
        "--context-limit",
        type=int,
        # The server's own --max-model-len. Knowing it up front is what lets a truncated program
        # be retried with the rest of the context instead of being cut in the same place three
        # times: a refusal states the limit, but a truncation states nothing at all.
        default=16384,
        help="The served model's context window, so a question can size its own budget",
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

    # What the run is about, and what this session will get through, are two different things.
    # `--id` narrows the run; `--limit` only says how far one Kaggle session expects to reach
    # before the twelve-hour cap. Folding the cap into the run's identity meant a first session
    # capped at 600 wrote a fingerprint no uncapped session could match, so the notebook meant
    # to finish the remaining questions refused the checkpoint it was handed.
    run_scope = _select_rows(
        _load_jsonl(args.retrieval),
        question_ids=args.question_ids,
        limit=None,
    )
    selected_rows = run_scope[: args.limit] if args.limit is not None else run_scope
    rows = _select_rows(
        selected_rows,
        question_ids=None,
        limit=None,
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
        context_limit=args.context_limit,
        execution_timeout=args.execution_timeout,
        request_timeout=args.request_timeout,
        memory_limit_mb=args.memory_limit_mb,
        thinking_mode=args.thinking_mode,
        max_attempts=args.max_attempts,
        selected_question_ids=[_as_int(row["id"], field="id") for row in run_scope],
        table_unit_source=args.table_unit_source,
        row_hierarchy=args.row_hierarchy,
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
        fallback_years: list[int] = []
        try:
            candidate_limit = _candidate_limit(row, minimum=args.candidate_tables)
            for table_ref in _as_str_list(row["fused"], field="fused"):
                if len(schemas) >= candidate_limit:
                    break
                record, table = store.load(str(table_ref), unit_source=args.table_unit_source)
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
            fallback_years = _as_int_list(spec.get("years"), field="query_spec.years")
            required_tickers = _as_str_list(spec.get("tickers"), field="query_spec.tickers")
            required_years = _as_int_list(spec.get("years"), field="query_spec.years")
            system, user = build_program_prompt(
                str(row["question"]),
                schemas,
                target_unit=str(spec["target_unit"]),
                target_divisor=float(spec["target_divisor"]),
                required_tickers=required_tickers,
                required_years=required_years,
                include_row_hierarchy=args.row_hierarchy,
            )
            successful: (
                tuple[dict[str, object], list[str], Dimension, str, float, list[str]] | None
            ) = None
            # Resized for this question whenever the server reveals what the prompt actually
            # costs. Measuring is arithmetic about the prompt, not a failed attempt, so it gets
            # its own small allowance: spending real attempts on it left question 213 with three
            # corrections, zero completions and nothing to show for the question.
            budget = _TokenBudget(args.max_tokens, args.context_limit)
            budget_corrections_left = 3
            attempt = 0
            while attempt < args.max_attempts:
                attempt += 1
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
                prompt_tokens: int | None = None
                attempt_max_tokens = budget.current
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
                        max_tokens=attempt_max_tokens,
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
                    prompt_tokens = getattr(usage, "prompt_tokens", None)
                    # The exact prompt length, which no refusal ever states exactly. Every later
                    # attempt on this question sizes itself from it.
                    budget.observe(prompt_tokens)
                    choice = response.choices[0]
                    content = choice.message.content
                    if not content:
                        raise ValueError("Model returned empty content")
                    if choice.finish_reason == "length":
                        raise _Truncated(
                            f"Model stopped at the {attempt_max_tokens}-token budget with the "
                            f"program unfinished"
                        )
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
                    query, answer, semantic_adjusted = normalize_absolute_difference(
                        question=str(row["question"]),
                        pandas_query=query,
                        answer=answer,
                    )
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
                    # Neither a refused request nor a truncated one is a wrong program: the first
                    # asked for more room than the prompt left, the second was given less than
                    # the answer needed. Both are arithmetic about the budget, so both get their
                    # own small allowance and neither spends one of the question's attempts.
                    resized = False
                    if isinstance(attempt_exc, BadRequestError):
                        resized = budget.shrink(str(attempt_exc))
                    elif isinstance(attempt_exc, _Truncated):
                        resized = budget.widen(prompt_tokens)
                    if resized and budget_corrections_left > 0:
                        budget_corrections_left -= 1
                        attempt_failures.pop()
                        attempt -= 1
                        continue
                    if isinstance(attempt_exc, BadRequestError) or attempt == args.max_attempts:
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
                "semantic_adjusted": semantic_adjusted,
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
            fallback = _fallback_program(
                frames=frames,
                target_divisor=fallback_divisor,
                question=str(row["question"]),
                years=fallback_years,
            )
            if fallback is not None:
                query, selected, answer = fallback
                query, answer, semantic_adjusted = normalize_absolute_difference(
                    question=str(row["question"]),
                    pandas_query=query,
                    answer=answer,
                )
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
                            "semantic_adjusted": semantic_adjusted,
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
