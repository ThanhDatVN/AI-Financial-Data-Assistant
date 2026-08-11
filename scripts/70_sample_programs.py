"""Build labelled training data backwards, from tables the corpus already holds.

The project cannot measure itself. Every question in `annotations/qrels_template.jsonl` is still
`unlabeled`, so "about 24% of the programs the model writes are correct" remains an estimate
drawn from an assumption, and every change to the answer branch costs twelve GPU hours plus a
submission before anyone learns whether it helped.

Going forwards -- question first, then program -- would need the model to write the program, and
that is the very thing under measurement. Going backwards does not: sample a program over a real
table, execute it, and the answer is right by construction. A separate step renders the Vietnamese
question afterwards, which is the one job an open model does reliably.

Nothing here calls a model. Nothing here needs a GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vifinqa.evidence.store import TableStore, parsed_table_to_long_frame  # noqa: E402
from vifinqa.indexing.manifest import ManifestRecord  # noqa: E402
from vifinqa.programs.compiler import compile_expression  # noqa: E402
from vifinqa.programs.executor import execute_expression  # noqa: E402
from vifinqa.programs.grounding import validate_answer_plausibility  # noqa: E402
from vifinqa.programs.ir import (  # noqa: E402
    ArgExtremumExpr,
    BinaryExpr,
    CellExpr,
    LiteralExpr,
    ScalarExpr,
)

# The question asks for a unit, and the program has to land in it. These are the divisors seen
# across the 1.012 real questions, with the phrasing the renderer will need later.
TARGET_UNITS: tuple[tuple[str, float], ...] = (
    ("đồng", 1.0),
    ("triệu đồng", 1e6),
    ("tỷ đồng", 1e9),
    ("trăm tỷ đồng", 1e11),
)

# A label that names no line item cannot anchor a question.
MIN_LABEL_CHARS = 12
# Base values are in dong. A denominator under a billion makes any ratio an artefact of rounding:
# the first smoke run produced a year-on-year change of 24.289.948% from a denominator of a few
# thousand dong.
MIN_DENOMINATOR = 1e9
# No real financial ratio or growth rate in this corpus runs past this, so anything that does is
# a parsing artefact rather than a fact worth asking about.
MAX_ABS_PERCENT = 500.0
# The largest balance sheet in this corpus totals around 2e15 dong, so a cell past 1e16 is not a
# figure. The ones that appear are rows naming a subsidiary in a related-party table, where OCR
# ran adjacent numbers together: the first dev set drew four of them, at 1.5e19 up to 2.1e78.
# Real values stop below 1e15 and the artefacts start above 1e19, so the cut is unambiguous.
MAX_ABS_BASE_VALUE = 1e16

# A row whose label opens with a bullet, a dash or a lone digit is a sub-item: it means nothing
# without the parent row above it, and a question naming it alone has no single answer.
_SUBITEM_PREFIXES = ("-", "+", "*", "•", "–", "(")
# Labels that name a movement line in an equity or provision roll-forward. They repeat across
# every column of the table, so they cannot identify one figure.
_ROLLFORWARD_MARKERS = ("số dư", "số cuối", "số đầu", "tại ngày", "cộng", "tổng cộng phát sinh")


def _normalise(label: str) -> str:
    """Fold a row label to something comparable across report years."""
    folded = unicodedata.normalize("NFKC", label).casefold().strip()
    return " ".join("".join(c for c in folded if c.isalnum() or c.isspace()).split())


# What one draw yields: the program, the unit its answer lands in, and the row and column it
# anchors on. Labels travel as plain strings because that is all the renderer ever needs.
type _Draw = tuple[ScalarExpr, str, str, str]


@dataclass(frozen=True, slots=True)
class Sample:
    """One labelled example: the program, what it computed, and where it came from."""

    family: str
    table_refs: tuple[str, ...]
    doc_ids: tuple[str, ...]
    ticker: str
    report_years: tuple[int, ...]
    scope: str
    row_label: str
    column_label: str
    section_title: str | None
    target_unit: str
    answer: float
    pandas_query: str
    program: dict[str, object]


def _expression_to_dict(expression: ScalarExpr) -> dict[str, object]:
    payload = asdict(expression)
    payload["node"] = type(expression).__name__
    return payload


def _section_key(title: str) -> str:
    """Fold a section heading to something that survives a change of report year.

    Notes are numbered, and the numbering moves: the same disclosure is "16 TIEN GUI VA VAY CAC
    TCTD KHAC" one year and "18 ..." the next. Matching the raw heading across years therefore
    almost always misses, which is what left the two multi-year families with no samples at all.
    """
    folded = _normalise(title)
    parts = folded.split()
    while parts and parts[0].isdigit():
        parts.pop(0)
    return " ".join(parts)


def _is_askable(label: str) -> bool:
    """Can a question name this row on its own and mean exactly one figure?"""
    stripped = label.strip()
    if len(stripped) < MIN_LABEL_CHARS or stripped.startswith(_SUBITEM_PREFIXES):
        return False
    folded = _normalise(stripped)
    if not folded or folded[0].isdigit():
        return False
    return not any(marker in folded for marker in _ROLLFORWARD_MARKERS)


def _current_period(frame: pd.DataFrame, report_year: int) -> pd.DataFrame:
    """Keep the columns that report the year the document is for.

    A statement carries the prior period beside the current one. Asking "in 2024" and reading the
    31.12.2023 column is a silently wrong label, and the first smoke run did exactly that.
    """
    header = frame["column_label"].astype(str)
    current = frame[header.str.contains(str(report_year), regex=False)]
    if current.empty:
        return current
    # Where several columns name the year, the leftmost is the primary one.
    return current[current["column_index"] == current["column_index"].min()]


def _numeric_cells(frame: pd.DataFrame, *, report_year: int | None = None) -> pd.DataFrame:
    """Rows a program can point at: a real number, under a label worth asking about."""
    # `base_value` holds None for unparsed cells, so the column is object dtype and `.abs()`
    # raises on it. Drop the blanks before doing any arithmetic.
    usable = frame[frame["base_value"].notna() & (frame["column_index"] > 0)]
    if usable.empty:
        return usable
    magnitude = usable["base_value"].astype(float).abs()
    usable = usable[(magnitude > 0) & (magnitude <= MAX_ABS_BASE_VALUE)]
    if usable.empty:
        return usable
    # `.map` over an empty frame returns an object-dtype series, and pandas reads an object-dtype
    # key as a list of column names rather than a mask -- which silently drops every column and
    # turns the next lookup into a KeyError. Casting to bool keeps it a mask either way.
    askable = usable["row_label"].astype(str).map(_is_askable).astype(bool)
    usable = usable[askable]
    if usable.empty or report_year is None:
        return usable
    return _current_period(usable, report_year)


def _execute(
    expression: ScalarExpr, frames: dict[str, pd.DataFrame], unit_name: str
) -> tuple[str, float] | None:
    """Compile and run, returning None for anything the project's own gates reject."""
    query = compile_expression(expression)
    try:
        answer = execute_expression(query, frames)
    except Exception:  # noqa: BLE001 - a sample that will not run is simply not a sample
        return None
    if not math.isfinite(answer):
        return None
    if unit_name == "phần trăm" and abs(answer) > MAX_ABS_PERCENT:
        return None
    try:
        validate_answer_plausibility(answer, expression)
    except Exception:  # noqa: BLE001
        return None
    return query, answer


def _sample_lookup(frame: pd.DataFrame, record: ManifestRecord, rng: random.Random) -> _Draw | None:
    """F1 -- read one cell and convert it into the unit the question will ask for."""
    usable = _numeric_cells(frame, report_year=int(record.report_year))
    if usable.empty:
        return None
    cell = usable.iloc[rng.randrange(len(usable))]
    unit_name, divisor = TARGET_UNITS[rng.randrange(len(TARGET_UNITS))]
    read = CellExpr(
        variable="df1",
        row_index=int(cell["row_index"]),
        column_index=int(cell["column_index"]),
    )
    expression: ScalarExpr = read
    if divisor != 1.0:
        expression = BinaryExpr(operator="/", left=read, right=LiteralExpr(value=divisor))
    return expression, unit_name, str(cell["row_label"]), str(cell["column_label"])


def _sample_ratio(frame: pd.DataFrame, record: ManifestRecord, rng: random.Random) -> _Draw | None:
    """F2 -- one line item as a percentage of another in the same column."""
    usable = _numeric_cells(frame, report_year=int(record.report_year))
    if len(usable) < 2:
        return None
    column = int(usable.iloc[rng.randrange(len(usable))]["column_index"])
    same_column = usable[usable["column_index"] == column]
    if len(same_column) < 2:
        return None
    # The denominator should be the larger figure, or the ratio is not a share of anything.
    ordered = same_column.reindex(same_column["base_value"].abs().sort_values().index)
    numerator = ordered.iloc[rng.randrange(len(ordered) - 1)]
    denominator = ordered.iloc[-1]
    if abs(float(denominator["base_value"])) < MIN_DENOMINATOR:
        return None
    if _normalise(str(numerator["row_label"])) == _normalise(str(denominator["row_label"])):
        return None
    expression = BinaryExpr(
        operator="*",
        left=BinaryExpr(
            operator="/",
            left=CellExpr(
                variable="df1",
                row_index=int(numerator["row_index"]),
                column_index=int(numerator["column_index"]),
            ),
            right=CellExpr(
                variable="df1",
                row_index=int(denominator["row_index"]),
                column_index=int(denominator["column_index"]),
            ),
        ),
        right=LiteralExpr(value=100.0),
    )
    return expression, "phần trăm", str(numerator["row_label"]), str(numerator["column_label"])


def _shared_label_rows(
    frames: dict[str, pd.DataFrame], rng: random.Random
) -> tuple[str, dict[str, pd.Series]] | None:
    """Find one line item that every year's table reports, matched on a folded label."""
    by_label: dict[str, dict[str, pd.Series]] = defaultdict(dict)
    for variable, frame in frames.items():
        year = int(frame["report_year"].iloc[0]) if not frame.empty else None
        usable = _numeric_cells(frame, report_year=year)
        for _, row in usable.iterrows():
            key = _normalise(str(row["row_label"]))
            # First column wins: later columns are prior-period restatements.
            if key and variable not in by_label[key]:
                by_label[key][variable] = row
    shared = [key for key, found in by_label.items() if len(found) == len(frames)]
    if not shared:
        return None
    chosen = shared[rng.randrange(len(shared))]
    return chosen, by_label[chosen]


def _sample_change(frames: dict[str, pd.DataFrame], rng: random.Random) -> _Draw | None:
    """F3 -- percentage change of one line item between two report years."""
    found = _shared_label_rows(frames, rng)
    if found is None:
        return None
    _, rows = found
    earlier, later = (rows[name] for name in sorted(rows))
    if abs(float(earlier["base_value"])) < MIN_DENOMINATOR:
        return None
    first = CellExpr(
        variable=str(earlier["variable"]),
        row_index=int(earlier["row_index"]),
        column_index=int(earlier["column_index"]),
    )
    second = CellExpr(
        variable=str(later["variable"]),
        row_index=int(later["row_index"]),
        column_index=int(later["column_index"]),
    )
    expression = BinaryExpr(
        operator="*",
        left=BinaryExpr(
            operator="/", left=BinaryExpr(operator="-", left=second, right=first), right=first
        ),
        right=LiteralExpr(value=100.0),
    )
    return expression, "phần trăm", str(later["row_label"]), str(later["column_label"])


def _sample_extremum(frames: dict[str, pd.DataFrame], rng: random.Random) -> _Draw | None:
    """F4 -- which year reported the largest (or smallest) value of one line item."""
    found = _shared_label_rows(frames, rng)
    if found is None:
        return None
    _, rows = found
    ordered = sorted(rows)
    values = tuple(
        CellExpr(
            variable=str(rows[name]["variable"]),
            row_index=int(rows[name]["row_index"]),
            column_index=int(rows[name]["column_index"]),
        )
        for name in ordered
    )
    years = tuple(LiteralExpr(value=float(rows[name]["report_year"])) for name in ordered)
    mode: Literal["argmin", "argmax"] = "argmax" if rng.random() < 0.5 else "argmin"
    expression = ArgExtremumExpr(mode=mode, keys=values, values=years)
    anchor = rows[ordered[-1]]
    return expression, "năm", str(anchor["row_label"]), str(anchor["column_label"])


def _frames_for(
    store: TableStore, refs: list[str]
) -> tuple[dict[str, pd.DataFrame], list[ManifestRecord]] | None:
    frames: dict[str, pd.DataFrame] = {}
    records: list[ManifestRecord] = []
    for position, ref in enumerate(refs, start=1):
        try:
            record, parsed = store.load(ref)
        except Exception:  # noqa: BLE001 - an unreadable table is skipped, not fatal
            return None
        frame = parsed_table_to_long_frame(record, parsed)
        # A table that parses to no rows yields an empty frame, and an empty frame carries no
        # columns at all -- every later `frame["base_value"]` then raises KeyError rather than
        # returning nothing. Reject it here so the failure stays local to one draw.
        if frame.empty:
            return None
        variable = f"df{position}"
        frame = frame.assign(variable=variable)
        frames[variable] = frame
        records.append(record)
    return frames, records


def _write(output: Path, kept: list[Sample]) -> None:
    """Persist what has been drawn so far, replacing the file atomically.

    A pool of several thousand takes hours, and holding all of it in memory until the last line
    means a run that dies at the two-hour mark leaves nothing behind. Writing through at
    intervals costs a fraction of a second and turns a total loss into a partial one.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_suffix(output.suffix + ".partial")
    with staging.open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(kept, start=1):
            handle.write(json.dumps({"id": index, **asdict(sample)}, ensure_ascii=False) + "\n")
    staging.replace(output)


def _usable_manifest(manifest: Path, min_rows: int, min_cols: int) -> pd.DataFrame:
    frame = pd.read_parquet(manifest)
    # A table whose unit was never resolved produces a number in an unknown scale, and a label
    # with the wrong scale is worse than no label at all.
    return frame[
        (frame["unit"] != "UNKNOWN")
        & (frame["n_rows"] >= min_rows)
        & (frame["n_cols"] >= min_cols)
        & frame["scope"].isin(["consolidated", "separate"])
    ].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data/processed/table_manifest.parquet"
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/ViFinQA")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1_000, help="How many samples to keep")
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--min-rows", type=int, default=3)
    parser.add_argument("--min-cols", type=int, default=2)
    parser.add_argument(
        "--tickers",
        help="Comma-separated tickers to draw from, so a split stays disjoint by company",
    )
    parser.add_argument(
        "--attempts-per-sample",
        type=int,
        default=12,
        help="Give up on a drawn table after this many failed program shapes",
    )
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be at least 1")

    rng = random.Random(args.seed)
    manifest = _usable_manifest(args.manifest, args.min_rows, args.min_cols)
    if args.tickers:
        wanted = {item.strip().upper() for item in args.tickers.split(",") if item.strip()}
        manifest = manifest[manifest["ticker"].isin(wanted)].reset_index(drop=True)
    if manifest.empty:
        parser.error("no usable tables after filtering")

    # Questions about one report far outnumber the rest, so single-table families dominate.
    families = (("lookup", 0.46), ("ratio", 0.22), ("change", 0.18), ("extremum", 0.14))
    by_document: dict[str, list[str]] = defaultdict(list)
    for ref, doc in zip(manifest["table_ref"], manifest["doc_id"], strict=True):
        by_document[doc].append(ref)
    by_ticker_year: dict[tuple[str, str], dict[int, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for ref, ticker, year, scope in zip(
        manifest["table_ref"],
        manifest["ticker"],
        manifest["report_year"],
        manifest["scope"],
        strict=True,
    ):
        by_ticker_year[(ticker, scope)][int(year)].append(ref)

    section_of = {
        str(ref): _section_key(str(title or ""))
        for ref, title in zip(manifest["table_ref"], manifest["section_title"], strict=True)
    }
    documents = sorted(by_document)
    multi_year = [key for key, years in by_ticker_year.items() if len(years) >= 2]
    # One store for the whole run: rebuilding it per draw re-reads the manifest every time.
    records = [ManifestRecord.from_dict(row) for row in manifest.to_dict("records")]
    store = TableStore(args.data_root, records)

    # Draw against a quota rather than a weighted coin. Weighting the draw skews the kept set
    # towards whichever family succeeds most often -- a first run asking for 46% lookups kept
    # 77% of them -- and the point of the set is to mirror the real question mix, not the
    # sampler's own hit rate.
    quota = {name: max(1, round(args.count * weight)) for name, weight in families}
    counts: dict[str, int] = defaultdict(int)
    kept: list[Sample] = []
    attempts = 0
    limit = args.count * args.attempts_per_sample
    while len(kept) < args.count and attempts < limit:
        attempts += 1
        short = [name for name, _ in families if counts[name] < quota[name]]
        if not short:
            break
        family = short[rng.randrange(len(short))]
        if family in {"lookup", "ratio"}:
            document = documents[rng.randrange(len(documents))]
            refs = [by_document[document][rng.randrange(len(by_document[document]))]]
        else:
            if not multi_year:
                continue
            key = multi_year[rng.randrange(len(multi_year))]
            years = sorted(by_ticker_year[key])
            span = 2 if family == "change" else min(len(years), rng.randint(3, 5))
            if len(years) < span:
                continue
            start = rng.randrange(len(years) - span + 1)
            chosen_years = years[start : start + span]
            # Drawing a table at random from each year almost never lands on the same statement
            # twice, so no line item is shared and the family yields nothing. Anchor on one year
            # and match the others by section title, which is what names the statement.
            anchor_year = chosen_years[0]
            anchor_pool = by_ticker_year[key][anchor_year]
            anchor = anchor_pool[rng.randrange(len(anchor_pool))]
            section = section_of.get(anchor)
            if not section:
                continue
            refs = [anchor]
            for year in chosen_years[1:]:
                matched = [
                    ref for ref in by_ticker_year[key][year] if section_of.get(ref) == section
                ]
                if not matched:
                    break
                refs.append(matched[rng.randrange(len(matched))])
            if len(refs) != span:
                continue
        loaded = _frames_for(store, refs)
        if loaded is None:
            continue
        frames, records = loaded
        if family == "lookup":
            drawn: _Draw | None = _sample_lookup(frames["df1"], records[0], rng)
        elif family == "ratio":
            drawn = _sample_ratio(frames["df1"], records[0], rng)
        elif family == "change":
            drawn = _sample_change(frames, rng)
        else:
            drawn = _sample_extremum(frames, rng)
        if drawn is None:
            continue
        expression, unit_name, row_label, column_label = drawn
        executed = _execute(expression, frames, unit_name)
        if executed is None:
            continue
        query, answer = executed
        first = records[0]
        counts[family] += 1
        if len(kept) and len(kept) % 250 == 0:
            _write(args.output, kept)
            print(
                f"  {len(kept)}/{args.count} kept after {attempts} attempts",
                flush=True,
            )
        kept.append(
            Sample(
                family=family,
                table_refs=tuple(refs),
                doc_ids=tuple(str(record.doc_id) for record in records),
                ticker=str(first.ticker),
                report_years=tuple(int(record.report_year) for record in records),
                scope=str(first.scope),
                row_label=row_label,
                column_label=column_label,
                section_title=first.section_title,
                target_unit=unit_name,
                answer=answer,
                pandas_query=query,
                program=_expression_to_dict(expression),
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write(args.output, kept)

    print(f"kept {len(kept)} samples from {attempts} attempts -> {args.output}")
    print("  by family: " + ", ".join(f"{name} {counts[name]}" for name in sorted(counts)))
    if len(kept) < args.count:
        print(f"  short of --count {args.count}: raise --attempts-per-sample or widen --tickers")


if __name__ == "__main__":
    main()
