"""Ask a candidate grammar policy the question that matters: what does it FORBID?

`program_grammar_for_target` is checked one way already -- a test proves every policy still leaves
all 499 gold roots reachable, so no constraint can make a correct answer impossible. That check is
necessary and it is not sufficient, and run dev_year paid a session to find out.

The policy under test there narrowed YEAR to `arg_extremum | select`. Every gold root stayed
reachable, so it shipped. It then blocked **0 of the 660 programs the model had actually got
wrong**, because `select` was among the two survivors and `select` is what the model writes. The
measured result was 0/198 on the extremum family: exactly unchanged.

So point the same failures at the candidate. A policy that forbids none of them cannot help, and
that is knowable in seconds from a rejection file that already exists. What a policy does forbid is
its ceiling, not its yield -- the model will emit something else, and whether the something else is
right is what the GPU session is for.

    python scripts/75_grammar_blocks_failures.py TK2/dev_year_rejected.s*.txt \\
        --programs outputs/synthetic/dev_programs.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vifinqa.programs.serde import (  # noqa: E402
    ROOT_GRAMMAR_POLICIES,
    program_grammar_for_target,
)


def _root_kinds(target_unit: str, policy: str) -> list[str]:
    node = program_grammar_for_target(target_unit, policy=policy)["properties"]["program"]  # type: ignore[index]
    refs = node.get("oneOf", [node])
    return [str(ref["$ref"]).rsplit("/", 1)[1] for ref in refs]


def _admits(target_unit: str, policy: str, kind: str) -> bool:
    names = _root_kinds(target_unit, policy)
    if names == ["expression_5"]:
        return True
    return any(name == kind or name.startswith(f"{kind}_") for name in names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rejected", type=Path, nargs="+", help="*_rejected.s*.jsonl from 73")
    parser.add_argument(
        "--programs",
        type=Path,
        required=True,
        help="the sampler file the questions came from, for each sample's target_unit",
    )
    args = parser.parse_args()

    units = {
        int(json.loads(line)["id"]): str(json.loads(line)["target_unit"])
        for line in args.programs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    failures: list[tuple[str, str, str]] = []
    without_log = 0
    for path in args.rejected:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            log = row.get("attempts_log")
            if not log:
                without_log += 1
                continue
            for attempt in log:
                program = attempt.get("program") or {}
                kind = program.get("kind")
                if kind:
                    failures.append((units[int(row["id"])], str(kind), str(row.get("family", "?"))))

    if not failures:
        raise SystemExit(
            "No programs recorded. Rejection files written before `attempts_log` existed "
            f"carry only the verdict, so there is nothing to project ({without_log} such rows)."
        )

    print(f"{len(failures)} programs the model got wrong, projected through each policy:\n")
    print(f"{'policy':16s} {'still legal':>12s} {'forbidden':>10s}   forbidden in")
    for policy in ROOT_GRAMMAR_POLICIES:
        blocked = [f for f in failures if not _admits(f[0], policy, f[1])]
        families = collections.Counter(family for _, _, family in blocked)
        share = len(blocked) / len(failures)
        print(
            f"{policy:16s} {len(failures) - len(blocked):5d}/{len(failures):<6d} "
            f"{len(blocked):6d} {share:4.0%}   {dict(families) or '-'}"
        )
    print(
        "\nA policy forbidding none of these cannot change the outcome. One that forbids many has "
        "only earned a session, not a result."
    )


if __name__ == "__main__":
    main()
