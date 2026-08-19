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
import random
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vifinqa.eval.metrics import answer_is_correct  # noqa: E402
from vifinqa.evidence.store import TableStore, parsed_table_to_long_frame  # noqa: E402
from vifinqa.generation.budget import (  # noqa: E402
    measure_prompt,
    room_for_output,
    tokenize_url,
)
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
from vifinqa.programs.serde import (  # noqa: E402
    ROOT_GRAMMAR_POLICIES,
    expression_from_dict,
    program_grammar_for_target,
)
from vifinqa.programs.year_answer import retarget_year_answer  # noqa: E402

SEED = 20260811


def _distractor_refs(
    store: TableStore, refs: list[str], count: int, rng: random.Random
) -> list[str]:
    """Other tables from the same reports, to measure at the width production actually shows.

    With gold tables alone, this script measures X with the retrieval difficulty taken out: a
    solver that only has to read the right table is not the solver that has to find it among
    twenty. That number is the honest one for comparing two prompts, and an upper bound on the
    one the scoreboard sees. Padding the list closes the gap at the cost of a longer prefill.

    The same reports rather than the whole corpus, because that is what routing already narrows
    the candidates to before ranking ever runs.
    """
    if count <= 0:
        return []
    gold = set(refs)
    documents = {ref.split("|", 1)[0] for ref in refs}
    pool = sorted(
        ref
        for ref, record in store.records.items()
        if record.doc_id in documents and ref not in gold
    )
    rng.shuffle(pool)
    return pool[:count]


def _candidates(
    store: TableStore, refs: list[str], unit_source: str, distractors: list[str], rng: random.Random
) -> tuple[list[CandidateSchema], dict[str, pd.DataFrame], set[str]]:
    """Load the gold tables under the variable names the sampler used: df1, df2, ... in order.

    With distractors, the order is drawn instead and the names follow the drawn positions. Both
    halves matter: leaving gold first would let the model read position instead of the question,
    and leaving gold named df1..dfk would let it read the name. Renaming is safe because nothing
    here re-executes the recorded program -- only the answer it produced is compared.

    The third return value names the gold variables, because `_fit` may only drop the others.
    """
    ordered = list(refs)
    if distractors:
        ordered = [*refs, *distractors]
        rng.shuffle(ordered)
    schemas: list[CandidateSchema] = []
    frames: dict[str, pd.DataFrame] = {}
    gold: set[str] = set()
    for ref in ordered:
        record, table = store.load(str(ref), unit_source=unit_source)
        frame = parsed_table_to_long_frame(record, table)
        numeric_cells = numeric_cells_of(frame)
        # Production skips a candidate that parses to no number rather than spend prompt budget
        # on it, and a distractor that behaves differently would not be measuring production.
        if ref not in refs and not numeric_cells:
            continue
        variable = f"df{len(schemas) + 1}"
        schemas.append(CandidateSchema(variable, record, table, numeric_cells))
        frames[variable] = frame
        if ref in refs:
            gold.add(variable)
    return schemas, frames, gold


def _fit(
    schemas: list[CandidateSchema],
    gold: set[str],
    *,
    render: Callable[[list[CandidateSchema]], tuple[str, str]],
    measure: Callable[[list[dict[str, str]]], int],
    context_limit: int,
    max_tokens: int,
) -> tuple[str, str, list[CandidateSchema]]:
    """Drop distractors -- never gold -- until the answer has somewhere to go.

    Production does this too, and without it a distractor measurement is not the one it claims to
    be. Measured over the 499 dev samples at 19 distractors, the median prompt reaches about 6,900
    tokens while the longest reaches 77,000: 23 leave no room for a 6,144-token answer and the
    worst are refused outright by a 16,384 context. Counting those as wrong answers would push X
    down by several points, and that number decides whether to rent a GPU.

    Gold stays because the answer depends on it: a sample whose gold table was dropped is not a
    harder question, it is a different one.
    """
    system, user = render(schemas)
    while True:
        measured = measure(
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
        )
        # No tokenizer route means no measurement, and guessing is what this replaced.
        if not measured or room_for_output(context_limit, measured) >= max_tokens:
            return system, user, schemas
        droppable = [index for index, schema in enumerate(schemas) if schema.variable not in gold]
        if not droppable:
            return system, user, schemas
        # The order was already drawn, so the last droppable one is an unbiased choice.
        schemas = [schema for index, schema in enumerate(schemas) if index != droppable[-1]]
        system, user = render(schemas)


def _solve(
    client: OpenAI,
    args: argparse.Namespace,
    system: str,
    user: str,
    frames: dict[str, pd.DataFrame],
    target_unit: str,
    target_divisor: float,
    attempt: int,
    record: dict[str, object],
    variable_years: dict[str, int],
) -> float | None:
    """One independent try.

    Any failure counts as a miss rather than an error: the sample is what is on trial here, and
    a question whose program will not compile is exactly what this script exists to find.

    `record` is filled in as the attempt proceeds rather than returned, so a raised exception
    still leaves behind the program that caused it. Without that the rejection file said only
    `solved_attempts: 0` and the next diagnosis had to start from a rerun.
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
                # Narrowed by the unit the answer has to be in, when the policy asks for it.
                "schema": program_grammar_for_target(target_unit, policy=args.root_grammar),
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
    record["program"] = program.get("program")
    expression = expression_from_dict(program.get("program"))
    if args.repair_year_answer:
        expression, repaired = retarget_year_answer(
            expression, target_unit=target_unit, variable_years=variable_years
        )
        if repaired:
            record["year_answer_repaired"] = True
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
    parser.add_argument(
        "--context-limit",
        type=int,
        default=16384,
        help="The served model's context window, so a padded prompt can be cut back to fit",
    )
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--execution-timeout", type=float, default=20.0)
    parser.add_argument("--memory-limit-mb", type=int, default=2048)
    parser.add_argument(
        "--attempts",
        type=int,
        default=3,
        help="Independent tries per question; one hit keeps the sample",
    )
    parser.add_argument(
        "--distractors",
        type=int,
        default=0,
        help=(
            "Pad the prompt with this many non-gold tables from the same reports. 0 measures the "
            "solver with the retrieval difficulty removed, which is an upper bound on the "
            "scoreboard's X; about 19 measures it at the width production shows, for a longer "
            "prefill per question"
        ),
    )
    parser.add_argument(
        "--root-grammar",
        default="off",
        choices=sorted(ROOT_GRAMMAR_POLICIES),
        help=(
            "Narrow the program's ROOT node to the shapes the target unit admits. 'off' is "
            "today's behaviour. Measured 19/08: with gold tables in the prompt, every family "
            "whose answer is a currency amount scored 0.84 or 0.13 while PERCENT scored 0.03 "
            "and 0.00 and YEAR scored 0.00, and `arg_extremum` was never emitted at all"
        ),
    )
    parser.add_argument(
        "--worked-example",
        action="store_true",
        help=(
            "Show one worked program of the shape this target expects. The renderer needed the "
            "same thing at the other end of the pipeline: told the rule twice it kept writing "
            "statements, and only a worked example moved it from 83 to 100 percent"
        ),
    )
    parser.add_argument(
        "--repair-year-answer",
        action="store_true",
        help=(
            "When the target is a YEAR and the program ranks the right cells but answers "
            "with one of them, hand back the year that cell belongs to. Measured 19/08: all "
            "198 extremum attempts were a select with the right operator and matching keys, "
            "and all 198 answered with the amount; 190 died saying exactly that"
        ),
    )
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.attempts < 1:
        parser.error("--attempts must be at least 1")
    if args.distractors < 0:
        parser.error("--distractors cannot be negative")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be in [0, --shard-count)")

    rows = [
        json.loads(line)
        for line in args.questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        rows = rows[: args.limit]
    # Limit first, then shard, exactly as 50_generate_programs.py orders them: `--limit` says how
    # far this experiment reaches, sharding only says who does which part of it. One client at a
    # time left a two-GPU server running eight sequences wide at one, which is why a 499-sample
    # measurement cost 9.4 hours instead of about one.
    rows = [
        row for position, row in enumerate(rows) if position % args.shard_count == args.shard_index
    ]

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
    tokenize = tokenize_url(args.base_url)
    print(f"filtering {len(rows)} questions, {len(done)} already judged", flush=True)

    misses: dict[str, int] = defaultdict(int)
    solved_by_family: dict[str, list[int]] = defaultdict(list)
    kept = 0
    greedy_hits = 0
    greedy_judged = 0
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
                    # Seeded per question, so the drawn distractors and their order are the same
                    # on a resumed session as on the one that started it.
                    rng = random.Random(args.seed + int(row["id"]))
                    schemas, frames, gold = _candidates(
                        store,
                        refs,
                        args.table_unit_source,
                        _distractor_refs(store, refs, args.distractors, rng),
                        rng,
                    )

                    def render(
                        candidates: list[CandidateSchema],
                        _row: dict[str, object] = row,
                        _unit: str = target_unit,
                        _divisor: float = target_divisor,
                    ) -> tuple[str, str]:
                        return build_program_prompt(
                            str(_row["question"]),
                            candidates,
                            target_unit=_unit,
                            target_divisor=_divisor,
                            required_tickers=[str(_row["ticker"])],
                            required_years=[int(year) for year in _row["report_years"]],
                            worked_example=args.worked_example,
                        )

                    system, user, schemas = _fit(
                        schemas,
                        gold,
                        render=render,
                        measure=lambda messages: measure_prompt(
                            tokenize, args.model, messages, args.request_timeout
                        ),
                        context_limit=args.context_limit,
                        max_tokens=args.max_tokens,
                    )
                    frames = {
                        schema.variable: frames[schema.variable]
                        for schema in schemas
                        if schema.variable in frames
                    }
                    # Which report year sits behind each variable. Only the repair needs it,
                    # so it is built here rather than threaded through prepare_program, which
                    # has no business knowing about manifests.
                    variable_years = {
                        schema.variable: int(schema.record.report_year) for schema in schemas
                    }
                except Exception as error:  # noqa: BLE001 - one lost sample, not a lost run
                    misses[f"setup:{type(error).__name__}"] += 1
                    continue

                expected = float(row["answer"])
                hits = 0
                # One entry per attempt, in order. Attempt 1 is greedy at temperature 0 and
                # mirrors production exactly; every later one samples at 0.7. Summing them into
                # a single count made `solved / attempts` a blend of two decoding regimes, and
                # the production-comparable figure -- how often the first, greedy try was right
                # -- could not be recovered once the session was over. That figure is what
                # decides whether renting a GPU buys anything, and re-measuring it costs a
                # whole session, so it is kept per attempt rather than added up.
                attempt_hits: list[int] = []
                attempts_log: list[dict[str, object]] = []
                for attempt in range(1, args.attempts + 1):
                    record: dict[str, object] = {"attempt": attempt}
                    try:
                        actual = _solve(
                            client,
                            args,
                            system,
                            user,
                            frames,
                            target_unit,
                            target_divisor,
                            attempt,
                            record,
                            variable_years,
                        )
                    except Exception as error:  # noqa: BLE001 - a miss is the measurement
                        misses[type(error).__name__] += 1
                        attempt_hits.append(0)
                        record["error"] = f"{type(error).__name__}: {error}"[:400]
                        attempts_log.append(record)
                        continue
                    record["answer"] = actual
                    if actual is not None and answer_is_correct(expected, actual):
                        hits += 1
                        attempt_hits.append(1)
                    else:
                        misses["wrong_answer"] += 1
                        attempt_hits.append(0)
                        record["error"] = "wrong_answer"
                    attempts_log.append(record)

                # Added beside `solved_attempts` rather than replacing it, so a session resumed
                # against rows an older revision wrote still reads and appends to the same file.
                verdict = {
                    **row,
                    "solved_attempts": hits,
                    "of_attempts": args.attempts,
                    "attempt_hits": attempt_hits,
                }
                # Only on the rejects, and only there: counting a gate is not the same as
                # knowing why it fired, and a solved row needs no post-mortem. Keeping the
                # program the model actually wrote is what turns "change scored 0/116" from a
                # number into something readable.
                if not hits:
                    verdict["attempts_log"] = attempts_log
                greedy_judged += 1
                greedy_hits += attempt_hits[0]
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
    if greedy_judged:
        # The figure to carry to the scoreboard comparison: one try, greedy, as production does
        # it. pass@k is what this filter keeps, and it is always the higher number.
        print(
            "  X pass@1 (attempt 1 only, greedy, mirrors production): "
            f"{greedy_hits / greedy_judged:.4f} over {greedy_judged} questions"
        )
    for family in sorted(solved_by_family):
        scores = solved_by_family[family]
        once = sum(1 for score in scores if score == 1)
        print(f"  {family:12s} kept {len(scores):4d}, solved exactly once {once:4d}")
    if misses:
        print("  misses: " + ", ".join(f"{name} {count}" for name, count in sorted(misses.items())))
    # The only breakdown of *how* X is lost, and it used to exist solely in this process's stdout.
    # A session whose notebook version was not saved took it to the grave.
    summary_path = args.output.with_name(args.output.stem + "_misses.json")
    summary_path.write_text(
        json.dumps(
            {
                "root_grammar": args.root_grammar,
                "worked_example": args.worked_example,
                "distractors": args.distractors,
                "attempts": args.attempts,
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "judged": greedy_judged,
                "kept": kept,
                "greedy_hits": greedy_hits,
                "misses": dict(sorted(misses.items())),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("  miss taxonomy ->", summary_path)


if __name__ == "__main__":
    main()
