"""Measure vLLM generation with and without JSON-schema constrained decoding.

This is a diagnostic, not an answer run.  It builds the exact production prompts for a small
fixed ID set, streams each response to separate time-to-first-token from decode time, and runs
the two modes in reversed order on alternating repeats to reduce warm-cache/order bias.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vifinqa.evidence.store import TableStore, parsed_table_to_long_frame  # noqa: E402
from vifinqa.generation.prompt import (  # noqa: E402
    CandidateSchema,
    build_program_prompt,
    numeric_cells_of,
)
from vifinqa.programs.serde import PROGRAM_GRAMMAR_SCHEMA  # noqa: E402

Mode = Literal["json_schema", "unconstrained"]


@dataclass(frozen=True, slots=True)
class Prompt:
    question_id: int
    system: str
    user: str


@dataclass(frozen=True, slots=True)
class Measurement:
    question_id: int
    repeat: int
    mode: Mode
    wall_seconds: float
    ttft_seconds: float | None
    decode_seconds: float | None
    completion_tokens: int | None
    decode_tokens_per_second: float | None
    content_characters: int
    valid_json: bool
    error: str | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _as_list(value: object, *, field: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a list")
    return value


def _as_int(value: object, *, field: str) -> int:
    if not isinstance(value, int | str) or isinstance(value, bool):
        raise TypeError(f"{field} must be an integer")
    return int(value)


def _candidate_limit(row: dict[str, object], minimum: int) -> int:
    spec = row.get("query_spec")
    if not isinstance(spec, dict):
        raise TypeError("query_spec must be an object")
    tickers = _as_list(spec.get("tickers"), field="query_spec.tickers")
    years = _as_list(spec.get("years"), field="query_spec.years")
    return max(minimum, max(1, len(tickers)) * max(1, len(years)))


def _build_prompts(
    rows: list[dict[str, object]],
    *,
    store: TableStore,
    candidate_tables: int,
    table_unit_source: str,
) -> list[Prompt]:
    prompts: list[Prompt] = []
    for row in rows:
        schemas: list[CandidateSchema] = []
        for table_ref in _as_list(row.get("fused"), field="fused"):
            if len(schemas) >= _candidate_limit(row, candidate_tables):
                break
            record, table = store.load(str(table_ref), unit_source=table_unit_source)
            frame = parsed_table_to_long_frame(record, table)
            numeric_cells = numeric_cells_of(frame)
            if not numeric_cells:
                continue
            schemas.append(CandidateSchema(f"df{len(schemas) + 1}", record, table, numeric_cells))
        if not schemas:
            raise ValueError(f"No numeric candidate table for id={row.get('id')}")
        spec = row.get("query_spec")
        if not isinstance(spec, dict):
            raise TypeError("query_spec must be an object")
        tickers = [str(item) for item in _as_list(spec.get("tickers"), field="tickers")]
        years = [
            _as_int(item, field="years item") for item in _as_list(spec.get("years"), field="years")
        ]
        system, user = build_program_prompt(
            str(row["question"]),
            schemas,
            target_unit=str(spec["target_unit"]),
            target_divisor=float(spec["target_divisor"]),
            required_tickers=tickers,
            required_years=years,
        )
        prompts.append(Prompt(_as_int(row["id"], field="id"), system, user))
    return prompts


def _stream_request(
    client: OpenAI,
    prompt: Prompt,
    *,
    model: str,
    mode: Mode,
    max_tokens: int,
    repeat: int,
) -> Measurement:
    started = time.perf_counter()
    first_content_at: float | None = None
    completion_tokens: int | None = None
    content_parts: list[str] = []
    try:
        if mode == "json_schema":
            stream = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt.system},
                    {"role": "user", "content": prompt.user},
                ],
                temperature=0.0,
                seed=20260802,
                max_tokens=max_tokens,
                stream=True,
                stream_options={"include_usage": True},
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "pandas_program",
                        "strict": True,
                        "schema": PROGRAM_GRAMMAR_SCHEMA,
                    },
                },
                extra_body={
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
        else:
            stream = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt.system},
                    {"role": "user", "content": prompt.user},
                ],
                temperature=0.0,
                seed=20260802,
                max_tokens=max_tokens,
                stream=True,
                stream_options={"include_usage": True},
                extra_body={
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
        for chunk in stream:
            now = time.perf_counter()
            usage = getattr(chunk, "usage", None)
            if usage is not None and getattr(usage, "completion_tokens", None) is not None:
                completion_tokens = int(usage.completion_tokens)
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            piece = getattr(choices[0].delta, "content", None)
            if piece:
                if first_content_at is None:
                    first_content_at = now
                content_parts.append(str(piece))
        finished = time.perf_counter()
        content = "".join(content_parts)
        try:
            json.loads(content)
            valid_json = True
        except json.JSONDecodeError:
            valid_json = False
        ttft = first_content_at - started if first_content_at is not None else None
        decode = finished - first_content_at if first_content_at is not None else None
        decode_rate = None
        if completion_tokens is not None and decode is not None and decode > 0:
            decode_rate = max(0, completion_tokens - 1) / decode
        return Measurement(
            question_id=prompt.question_id,
            repeat=repeat,
            mode=mode,
            wall_seconds=finished - started,
            ttft_seconds=ttft,
            decode_seconds=decode,
            completion_tokens=completion_tokens,
            decode_tokens_per_second=decode_rate,
            content_characters=len(content),
            valid_json=valid_json,
            error=None,
        )
    except Exception as error:  # noqa: BLE001 - diagnostics must retain every failed request
        finished = time.perf_counter()
        return Measurement(
            question_id=prompt.question_id,
            repeat=repeat,
            mode=mode,
            wall_seconds=finished - started,
            ttft_seconds=None,
            decode_seconds=None,
            completion_tokens=completion_tokens,
            decode_tokens_per_second=None,
            content_characters=sum(map(len, content_parts)),
            valid_json=False,
            error=f"{type(error).__name__}: {error}",
        )


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _summary(rows: list[Measurement], mode: Mode) -> dict[str, object]:
    selected = [row for row in rows if row.mode == mode]
    rates = [row.decode_tokens_per_second for row in selected]
    ttfts = [row.ttft_seconds for row in selected]
    return {
        "mode": mode,
        "requests": len(selected),
        "successful": sum(row.error is None for row in selected),
        "valid_json": sum(row.valid_json for row in selected),
        "median_wall_seconds": _median([row.wall_seconds for row in selected]),
        "median_ttft_seconds": _median([value for value in ttfts if value is not None]),
        "median_decode_tokens_per_second": _median(
            [value for value in rates if value is not None and math.isfinite(value)]
        ),
    }


def _comparison(summaries: list[dict[str, object]]) -> dict[str, object]:
    by_mode = {str(summary["mode"]): summary for summary in summaries}
    constrained = by_mode["json_schema"]
    unconstrained = by_mode["unconstrained"]
    constrained_rate = constrained["median_decode_tokens_per_second"]
    unconstrained_rate = unconstrained["median_decode_tokens_per_second"]
    speed_ratio: float | None = None
    if (
        isinstance(constrained_rate, int | float)
        and isinstance(unconstrained_rate, int | float)
        and constrained_rate > 0
    ):
        speed_ratio = float(unconstrained_rate) / float(constrained_rate)

    successful = unconstrained["successful"]
    valid_json = unconstrained["valid_json"]
    unconstrained_valid_json_rate: float | None = None
    if isinstance(successful, int) and successful > 0 and isinstance(valid_json, int):
        unconstrained_valid_json_rate = valid_json / successful

    if speed_ratio is None:
        diagnosis = "insufficient_measurements"
    elif speed_ratio >= 1.5:
        diagnosis = "guided_decoding_is_a_likely_bottleneck"
    elif speed_ratio <= 1.2:
        diagnosis = "guided_decoding_is_not_the_primary_bottleneck"
    else:
        diagnosis = "mixed_result_repeat_before_changing_decoding"
    return {
        "unconstrained_over_json_schema_decode_speed": speed_ratio,
        "unconstrained_valid_json_rate": unconstrained_valid_json_rate,
        "diagnosis": diagnosis,
        "thresholds": {"likely_bottleneck": 1.5, "not_primary": 1.2},
        "warning": (
            "This diagnosis measures throughput only. Never use unconstrained output for a "
            "submission until schema validation and answer execution pass."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--project-revision")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key-env", default="VLLM_API_KEY")
    parser.add_argument("--candidate-tables", type=int, default=20)
    parser.add_argument(
        "--table-unit-source",
        choices=("manifest", "latest"),
        default="latest",
        help="Unit fallback policy used by the production prompt builder.",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--request-timeout", type=float, default=360.0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--id", dest="question_ids", action="append", required=True, type=int)
    args = parser.parse_args()
    if (
        min(
            args.candidate_tables,
            args.max_tokens,
            args.request_timeout,
            args.repeats,
            args.concurrency,
        )
        <= 0
    ):
        parser.error("numeric limits must be positive")
    if len(args.question_ids) != len(set(args.question_ids)):
        parser.error("--id values must be unique")

    retrieval = _load_jsonl(args.retrieval)
    by_id = {_as_int(row["id"], field="id"): row for row in retrieval}
    missing = [question_id for question_id in args.question_ids if question_id not in by_id]
    if missing:
        parser.error(f"unknown question IDs: {missing}")
    selected = [by_id[question_id] for question_id in args.question_ids]
    candidate_refs = {
        str(ref) for row in selected for ref in _as_list(row.get("fused"), field="fused")
    }
    store = TableStore.from_parquet(args.data_root, args.manifest, candidate_refs)
    prompts = _build_prompts(
        selected,
        store=store,
        candidate_tables=args.candidate_tables,
        table_unit_source=args.table_unit_source,
    )
    client = OpenAI(
        base_url=args.base_url,
        api_key=os.environ.get(args.api_key_env, "local-vllm"),
        timeout=args.request_timeout,
        max_retries=0,
    )

    measurements: list[Measurement] = []
    for repeat in range(args.repeats):
        modes: tuple[Mode, Mode] = (
            ("json_schema", "unconstrained")
            if repeat % 2 == 0
            else ("unconstrained", "json_schema")
        )
        for mode in modes:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = [
                    pool.submit(
                        _stream_request,
                        client,
                        prompt,
                        model=args.model,
                        mode=mode,
                        max_tokens=args.max_tokens,
                        repeat=repeat,
                    )
                    for prompt in prompts
                ]
                measurements.extend(future.result() for future in futures)

    summary_modes: tuple[Mode, Mode] = ("json_schema", "unconstrained")
    summaries = [_summary(measurements, mode) for mode in summary_modes]
    payload = {
        "model": args.model,
        "model_revision": args.model_revision,
        "project_revision": args.project_revision,
        "retrieval_sha256": _sha256(args.retrieval),
        "manifest_sha256": _sha256(args.manifest),
        "table_unit_source": args.table_unit_source,
        "candidate_tables": args.candidate_tables,
        "max_tokens": args.max_tokens,
        "repeats": args.repeats,
        "concurrency": args.concurrency,
        "question_ids": args.question_ids,
        "summary": summaries,
        "comparison": _comparison(summaries),
        "measurements": [asdict(row) for row in measurements],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
