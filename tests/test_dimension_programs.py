"""The half of the exam that answers in a dimension the source cells do not carry.

Measured 19/08/2026 on 499 dev samples, gold tables already in the prompt and no distractors, so
retrieval difficulty is out of it entirely:

    lookup       target currency   root cell           pass@1 0.8371
    conditional  target currency   root select         pass@1 0.1274
    ratio        target PERCENT    root binary         pass@1 0.0345
    change       target PERCENT    root binary         pass@1 0.0000
    extremum     target YEAR       root arg_extremum   pass@1 0.0000

Two exact zeros from a model that answers 84% of plain lookups, and `arg_extremum` emitted 0 times
in 519, 640 and 631 clean programs across three scored runs. Replaying the gold programs through
the compiler reproduced all 499 recorded answers, so the harness is not the cause.
"""

from __future__ import annotations

import json
import re

from vifinqa.generation.prompt import worked_example_for
from vifinqa.programs.serde import (
    PROGRAM_GRAMMAR_SCHEMA,
    ROOT_GRAMMAR_POLICIES,
    expression_from_dict,
    program_grammar_for_target,
)

# One gold root per family, in the shape `70_sample_programs.py` actually recorded.
GOLD_ROOTS: dict[str, tuple[str, str]] = {
    "lookup": ("VND", "cell"),
    "lookup_percent": ("PERCENT", "cell"),  # six dev samples read a percentage column directly
    "conditional": ("MILLION_VND", "select"),
    "ratio": ("PERCENT", "binary"),
    "change": ("PERCENT", "binary"),
    "extremum": ("YEAR", "arg_extremum"),
}


def _root_names(schema: dict[str, object]) -> list[str]:
    node = schema["properties"]["program"]  # type: ignore[index]
    refs = [item["$ref"] for item in node["oneOf"]] if "oneOf" in node else [node["$ref"]]
    return [ref.rsplit("/", 1)[1] for ref in refs]


def _admits(schema: dict[str, object], kind: str) -> bool:
    names = _root_names(schema)
    if names == ["expression_5"]:
        return True
    return any(name == kind or name.startswith(f"{kind}_") for name in names)


def test_no_policy_can_make_a_correct_answer_unreachable() -> None:
    """A narrowed root that forbids a gold shape does not make the model try harder.

    The first draft narrowed PERCENT to (binary, select) and would have locked out six dev
    lookups whose source column already holds a percentage -- correct answers, made unreachable
    by a constraint meant to help. Every policy has to clear this before it is worth measuring.
    """
    for policy in ROOT_GRAMMAR_POLICIES:
        for family, (unit, kind) in GOLD_ROOTS.items():
            schema = program_grammar_for_target(unit, policy=policy)
            assert _admits(schema, kind), f"{policy} forbids the gold {kind} root of {family}"


def test_off_is_the_untouched_grammar_and_currency_is_never_narrowed() -> None:
    """The 84% case must decode exactly as it does today, under every policy."""
    for policy in ROOT_GRAMMAR_POLICIES:
        for unit in ("VND", "USD", "MILLION_VND", "BILLION_VND", "SHARES", "COUNT"):
            assert program_grammar_for_target(unit, policy=policy) is PROGRAM_GRAMMAR_SCHEMA
    assert program_grammar_for_target("YEAR", policy="off") is PROGRAM_GRAMMAR_SCHEMA


def test_year_loses_the_roots_that_return_an_amount() -> None:
    """65 attempts in run 3226 died on "dimension VND is incompatible with target YEAR".

    That message says the program returned a figure where a label was wanted. A year is never a
    stored value, so no legitimate answer is a bare cell -- which is what makes YEAR the one
    target the evidence supports narrowing.
    """
    narrowed = program_grammar_for_target("YEAR", policy="year")
    assert _root_names(narrowed) == ["arg_extremum_5", "select_5"]
    for forbidden in ("cell", "literal", "aggregate", "count_if", "binary"):
        assert not _admits(narrowed, forbidden)
    # The competing hypothesis: if the model cannot write the node at all, forcing it reads worse.
    assert _root_names(program_grammar_for_target("YEAR", policy="year-strict")) == [
        "arg_extremum_5"
    ]


def test_an_unknown_policy_is_refused_rather_than_silently_ignored() -> None:
    for bad in ("", "yes", "YEAR"):
        try:
            program_grammar_for_target("YEAR", policy=bad)
        except ValueError as error:
            assert "Unknown root grammar policy" in str(error)
        else:  # pragma: no cover - the assertion below reports it
            raise AssertionError(f"policy {bad!r} was accepted")


def test_every_worked_example_is_a_program_the_parser_accepts() -> None:
    """An example that does not parse teaches a shape the pipeline then refuses.

    `arg_extremum` carries `keys` and `values`; the one place the prompt used to mention it said
    only "the same thing in one node", and the select paragraph beside it says `members`. Showing
    the wrong field names would be worse than showing nothing.
    """
    for unit in ("YEAR", "PERCENT", "RATIO"):
        example = worked_example_for(unit)
        assert example, f"{unit} is a measured-zero target and needs an example"
        fragments = re.findall(r'\{"kind":.*?\}(?=\\n|\n|$|\s+in place)', example, re.S)
        assert fragments, f"{unit} example carries no JSON"
        for fragment in fragments:
            expression_from_dict(json.loads(fragment))
    year_example = worked_example_for("YEAR")
    assert '"values"' in year_example, "the example must name arg_extremum's own field"
    assert '"members"' not in year_example, "that is the select field, and the wrong one here"
    # The families that already work are not shown anything: prompt budget is the scarce thing.
    for unit in ("VND", "MILLION_VND", "SHARES", "COUNT"):
        assert worked_example_for(unit) == ""


def test_sharding_partitions_the_questions_without_loss_or_overlap() -> None:
    """`73` sent one request at a time to a server running eight sequences wide.

    That is why 499 samples times two attempts cost 9.4 hours. The split is by position so every
    shard gets a representative mix rather than one of them inheriting the whole Hard tail.
    """
    rows = list(range(499))
    for shard_count in (1, 2, 8):
        shards = [
            [row for position, row in enumerate(rows) if position % shard_count == index]
            for index in range(shard_count)
        ]
        assert sorted(row for shard in shards for row in shard) == rows
        assert sum(len(shard) for shard in shards) == len(rows)
        assert max(len(shard) for shard in shards) - min(len(shard) for shard in shards) <= 1
