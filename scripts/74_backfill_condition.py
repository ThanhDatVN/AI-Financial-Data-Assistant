"""Recover the filter a Hard sample was built around, from the program it already carries.

`70_sample_programs.py` only started recording `condition` today. Every Hard sample drawn before
that -- 106 of the 499 dev programs and 165 of the 796 in the train pool -- names only the line
item that supplies the answer, not the one that decides which years count. Rendered from that
alone, the question asks the plainer thing, whose answer `_sample_conditional` had already proved
differs, so the whole 21.3% tier would be written fluently and then filtered out for being wrong.

Regenerating is not the fix. The sets were drawn on an earlier revision and the current sampler
does not reproduce them: `--split dev --count 499 --seed 20260811` returns 416 samples with 23
Hard ones, against the 106 the file holds, because later commits tightened multi-year matching.
Swapping a set that mirrors the paper's difficulty mix for one that does not, in order to add a
field, would cost more than it recovers.

The field is recoverable anyway. The program is in the file: its `conditions[0].left` cells name
the gate row by coordinate, and its `right` literal holds the threshold. Reading the gate's label
and counting the years that clear it is deterministic, needs no model, and leaves every other
field untouched -- so the set stays the one RUN-028 validated at 499/499 through grounding.

The threshold is stored but the phrasing uses the rank: it is an element of the gate series
itself, so every non-survivor sits at or below it, and "the N years where this was highest"
describes exactly the same years in the words a question would use.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vifinqa.evidence.store import TableStore, parsed_table_to_long_frame  # noqa: E402
from vifinqa.indexing.manifest import ManifestRecord  # noqa: E402
from vifinqa.parsing.normalize import ascii_compact  # noqa: E402


class Unrecoverable(ValueError):
    """This sample's filter cannot be read back, so the sample cannot be phrased."""


def _cells(condition: dict[str, object]) -> list[dict[str, object]]:
    left = condition.get("left")
    if not isinstance(left, list) or not left:
        raise Unrecoverable("condition has no left operands")
    return [cell for cell in left if isinstance(cell, dict)]


def _threshold(condition: dict[str, object]) -> float:
    right = condition.get("right")
    if not isinstance(right, dict) or "value" not in right:
        raise Unrecoverable("condition has no literal threshold")
    return float(right["value"])


def recover(row: dict[str, object], frames: dict[str, pd.DataFrame]) -> dict[str, object]:
    """Read the gate's label and count the years that clear it."""
    program = row.get("program")
    if not isinstance(program, dict) or program.get("kind") != "select":
        raise Unrecoverable("Hard samples are select nodes")
    conditions = program.get("conditions")
    if not isinstance(conditions, list) or len(conditions) != 1:
        raise Unrecoverable("expected exactly one condition")
    condition = conditions[0]
    if not isinstance(condition, dict):
        raise Unrecoverable("condition is not an object")
    threshold = _threshold(condition)
    labels: list[str] = []
    survivors = 0
    for cell in _cells(condition):
        frame = frames.get(str(cell["variable"]))
        if frame is None:
            raise Unrecoverable(f"program reads unknown variable {cell['variable']}")
        match = frame[
            (frame["row_index"] == int(cell["row_index"]))
            & (frame["column_index"] == int(cell["column_index"]))
        ]
        if match.empty:
            raise Unrecoverable("gate coordinate is not in its table")
        labels.append(str(match.iloc[0]["row_label"]).strip())
        value = match.iloc[0]["base_value"]
        if pd.isna(value):
            raise Unrecoverable("gate coordinate holds no number")
        if float(value) > threshold:
            survivors += 1
    # The sampler picks the gate rows by a shared label across years, and it compares them folded
    # -- so "Phải trả nhà cung cấp" and "▪ Phải trả nhà cung cấp" are one line item to it, and
    # three of the 106 dev samples differ only by a bullet or a footnote marker. Comparing the
    # surface forms would refuse those for untidiness; comparing them the sampler's way keeps the
    # check where it belongs, on coordinates that stopped pointing at the same row.
    if len({ascii_compact(label) for label in labels}) != 1:
        raise Unrecoverable(f"gate rows carry different labels: {sorted(set(labels))}")
    # Its other invariant: fewer than two survivors makes the ranking vacuous, and it refused to
    # emit such a draw. Seeing one now means the values changed underneath the program.
    if survivors < 2:
        raise Unrecoverable(f"only {survivors} year(s) clear the threshold")
    return {
        # The question names this row, so give it the tidiest of the surface forms rather than
        # whichever year happened to be first: a bullet or a footnote marker is table furniture.
        "row_label": min(labels, key=len),
        "top_n": survivors,
        "of_years": len(labels),
        "comparator": str(condition.get("comparator", ">")),
        "threshold": threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("programs", type=Path, help="JSONL from 70_sample_programs.py")
    parser.add_argument("--output", type=Path, help="Defaults to rewriting the input in place")
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data/processed/table_manifest.parquet"
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/ViFinQA")
    parser.add_argument("--table-unit-source", default="latest")
    args = parser.parse_args()
    output = args.output or args.programs

    rows = [
        json.loads(line)
        for line in args.programs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    hard = [row for row in rows if str(row.get("family")) == "conditional"]
    todo = [row for row in hard if not row.get("condition")]
    print(f"{len(rows)} samples, {len(hard)} Hard, {len(todo)} missing their filter")
    if not todo:
        print("nothing to do")
        return

    needed = {str(ref) for row in todo for ref in row["table_refs"]}
    manifest = pd.read_parquet(args.manifest)
    store = TableStore(
        args.data_root,
        [
            ManifestRecord.from_dict(record)
            for record in manifest[manifest["table_ref"].isin(needed)].to_dict("records")
        ],
    )
    cache: dict[str, pd.DataFrame] = {}

    def frame_of(ref: str) -> pd.DataFrame:
        if ref not in cache:
            record, table = store.load(ref, unit_source=args.table_unit_source)
            cache[ref] = parsed_table_to_long_frame(record, table)
        return cache[ref]

    recovered = 0
    failures: list[tuple[int, str]] = []
    for position, row in enumerate(todo, start=1):
        refs = [str(ref) for ref in row["table_refs"]]
        try:
            frames = {f"df{index}": frame_of(ref) for index, ref in enumerate(refs, start=1)}
            row["condition"] = recover(row, frames)
        except Exception as error:  # noqa: BLE001 - one unrecoverable sample, not a lost file
            failures.append((int(row["id"]), f"{type(error).__name__}: {error}"))
        else:
            recovered += 1
        if position % 25 == 0:
            print(f"  {position}/{len(todo)}", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"recovered {recovered}/{len(todo)} -> {output}")
    for sample_id, reason in failures[:10]:
        print(f"  id {sample_id}: {reason}")
    if failures:
        # A sample nobody can phrase is worse than a missing sample: it reads as a fluent question
        # with the wrong answer. 71_render_questions.py refuses these, so leaving them is safe.
        print(f"  {len(failures)} left without a filter; 71_render_questions.py will refuse them.")


if __name__ == "__main__":
    main()
