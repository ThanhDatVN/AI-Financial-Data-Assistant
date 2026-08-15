"""Keep the synthetic questions a solver can actually answer, and drop the rest.

`70_sample_programs.py` guarantees the answer is right for the program. Nothing guarantees the
question `71_render_questions.py` wrote asks for that answer. A question that names the wrong
year, omits the scope, or paraphrases a line item into something three other rows also match is
still fluent Vietnamese -- and training on it teaches a mapping the evidence does not support.

So put it to the test the same way the competition will: hand the question and its gold tables to
the shipped solver prompt and see whether the number comes back. `--attempts` independent tries
are allowed, and one hit is enough. Zero hits out of several means either the question is
ambiguous or the task is beyond the model, and neither is worth training on unsupervised.

The count of hits is kept as `solved_attempts`, because it is the only difficulty signal in the
set that was measured rather than assumed. A question solved on every attempt teaches little; one
solved on exactly one is where the learning is. Filtering on that is a later decision, so this
script records the number and refuses only the zeroes.

The solver here is `build_program_prompt` itself, not a copy of it. A filter that asks a
different question than the pipeline does would certify samples the pipeline still fails.

Only open weights, per the competition rules -- the same served model the run itself uses.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vifinqa.eval.metrics import answer_is_correct  # noqa: E402
from vifinqa.evidence.store import TableStore, parsed_table_to_long_frame  # noqa: E402
from vifinqa.generation.prompt import (  # noqa: E402
    CandidateSchema,
    build_program_prompt,
    numeric_cells_of,
)
from vifinqa.indexing.manifest import ManifestRecord  # noqa: E402
from vifinqa.programs.compiler import compile_expression  # noqa: E402
from vifinqa.programs.executor import execute_expression_isolated  # noqa: E402
from vifinqa.programs.grounding import (  # noqa: E402
    prepare_program,
    referenced_variables,
    validate_answer_plausibility,
)
from vifinqa.programs.serde import PROGRAM_GRAMMAR_SCHEMA, expression_from_dict  # noqa: E402

SEED = 20260811


def _candidates(
    store: TableStore, refs: list[str], unit_source: str
) -> tuple[list[CandidateSchema], dict[str, pd.DataFrame]]:
    """Load the gold tables under the variable names the sampler used: df1, df2, ... in order."""
    schemas: list[CandidateSchema] = []
    frames: dict[str, pd.DataFrame] = {}
    for ref in refs:
        record, table = store.load(str(ref), unit_source=unit_source)
        frame = parsed_table_to_long_frame(record, table)
        variable = f"df{len(schemas) + 1}"
        schemas.append(CandidateSchema(variable, record, table, numeric_cells_of(frame)))
        frames[variable] = frame
    return schemas, frames


def _solve(
    client: OpenAI,
    args: argparse.Namespace,
    system: str,
    user: str,
    frames: dict[str, pd.DataFrame],
    target_unit: str,
    target_divisor: float,
    attempt: int,
) -> float | None:
    """One independent try.

    Any failure counts as a miss rather than an error: the sample is what is on trial here, and
    a question whose program will not compile is exactly what this script exists to find.
    """
    extra_body: dict[str, object] = {"top_k": 20}
    if args.thinking_mode == "disabled":
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        # The first try mirrors production exactly; later ones vary, so a miss means the question
        # resists more than one reading rather than one unlucky decode.
        temperature=0.0 if attempt == 1 else 0.7,
        top_p=1.0 if attempt == 1 else 0.8,
        seed=args.seed + attempt - 1,
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
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Model returned empty content")
    program = json.loads(content)
    if not isinstance(program, dict):
        raise ValueError("Model output must be an object")
    expression = expression_from_dict(program.get("program"))
    selected = sorted(referenced_variables(expression))
    unknown = [variable for variable in selected if variable not in frames]
    if unknown:
        raise ValueError(f"Program references unknown variables: {unknown}")
    selected_frames = {variable: frames[variable] for variable in selected}
    prepared, _ = prepare_program(
        expression,
        selected_variables=selected,
        frames=selected_frames,
        target_unit=target_unit,
        target_divisor=target_divisor,
    )
    answer = execute_expression_isolated(
        compile_expression(prepared),
        selected_frames,
        timeout_seconds=args.execution_timeout,
        memory_limit_mb=args.memory_limit_mb,
    )
    validate_answer_plausibility(answer, expression)
    return float(answer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", type=Path, help="JSONL from 71_render_questions.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, help="Where to keep the misses, for inspection")
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data/processed/table_manifest.parquet"
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/ViFinQA")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key-env", default="VIFINQA_API_KEY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--table-unit-source", default="latest")
    parser.add_argument("--thinking-mode", default="disabled", choices=["disabled", "enabled"])
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--execution-timeout", type=float, default=20.0)
    parser.add_argument("--memory-limit-mb", type=int, default=2048)
    parser.add_argument(
        "--attempts",
        type=int,
        default=3,
        help="Independent tries per question; one hit keeps the sample",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.attempts < 1:
        parser.error("--attempts must be at least 1")

    rows = [
        json.loads(line)
        for line in args.questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        rows = rows[: args.limit]

    manifest = pd.read_parquet(args.manifest)
    store = TableStore(
        args.data_root, [ManifestRecord.from_dict(row) for row in manifest.to_dict("records")]
    )
    client = OpenAI(
        base_url=args.base_url,
        api_key=os.environ.get(args.api_key_env, "local-vllm"),
        timeout=args.request_timeout,
        max_retries=0,
    )

    # Filtering a few thousand questions outlasts a Kaggle session, so resume rather than pay
    # for the same verdicts twice.
    done: set[int] = set()
    for path in (args.output, args.rejected):
        if path and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    done.add(int(json.loads(line)["id"]))
    print(f"filtering {len(rows)} questions, {len(done)} already judged", flush=True)

    misses: dict[str, int] = defaultdict(int)
    solved_by_family: dict[str, list[int]] = defaultdict(list)
    kept = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rejected_handle = (
        args.rejected.open("a", encoding="utf-8") if args.rejected else None  # noqa: SIM115
    )
    try:
        with args.output.open("a", encoding="utf-8") as handle:
            for position, row in enumerate(rows, start=1):
                if int(row["id"]) in done:
                    continue
                refs = [str(ref) for ref in row["table_refs"]]
                target_unit = str(row["target_unit"])
                # Recorded by the sampler rather than re-derived: the divisor decides three
                # orders of magnitude, and a second table of them would be a second thing to
                # drift.
                target_divisor = float(row["target_divisor"])
                try:
                    schemas, frames = _candidates(store, refs, args.table_unit_source)
                    system, user = build_program_prompt(
                        str(row["question"]),
                        schemas,
                        target_unit=target_unit,
                        target_divisor=target_divisor,
                        required_tickers=[str(row["ticker"])],
                        required_years=[int(year) for year in row["report_years"]],
                    )
                except Exception as error:  # noqa: BLE001 - one lost sample, not a lost run
                    misses[f"setup:{type(error).__name__}"] += 1
                    continue

                expected = float(row["answer"])
                hits = 0
                for attempt in range(1, args.attempts + 1):
                    try:
                        actual = _solve(
                            client, args, system, user, frames, target_unit, target_divisor, attempt
                        )
                    except Exception as error:  # noqa: BLE001 - a miss is the measurement
                        misses[type(error).__name__] += 1
                        continue
                    if actual is not None and answer_is_correct(expected, actual):
                        hits += 1
                    else:
                        misses["wrong_answer"] += 1

                verdict = {**row, "solved_attempts": hits, "of_attempts": args.attempts}
                if hits:
                    handle.write(json.dumps(verdict, ensure_ascii=False) + "\n")
                    handle.flush()
                    kept += 1
                    solved_by_family[str(row.get("family", "?"))].append(hits)
                elif rejected_handle is not None:
                    rejected_handle.write(json.dumps(verdict, ensure_ascii=False) + "\n")
                    rejected_handle.flush()
                if position % 50 == 0:
                    print(f"  {position}/{len(rows)}, kept {kept}", flush=True)
    finally:
        if rejected_handle is not None:
            rejected_handle.close()

    print(f"kept {kept} of {len(rows) - len(done)} judged -> {args.output}")
    for family in sorted(solved_by_family):
        scores = solved_by_family[family]
        once = sum(1 for score in scores if score == 1)
        print(f"  {family:12s} kept {len(scores):4d}, solved exactly once {once:4d}")
    if misses:
        print("  misses: " + ", ".join(f"{name} {count}" for name, count in sorted(misses.items())))


if __name__ == "__main__":
    main()
