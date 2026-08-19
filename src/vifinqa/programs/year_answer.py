"""Turn "which of these amounts is largest" into "which year that amount belongs to".

Run dev_year measured the extremum family at 0 of 198 and, for the first time, kept the programs.
They are not confused. Every one of the 198 is a `select` with the right operator, with `keys` and
`members` the same length, ranking the right cells -- the model found the extreme correctly. It
answers with the amount it ranked by instead of the year that amount sits in, so grounding refuses
it for reporting VND where a YEAR was asked for. 190 of the 198 die on exactly that message.

The message has told it what to do since 37209f7 -- "make the members the year literals" -- and
across 198 attempts it complied zero times. Narrowing the grammar to `arg_extremum | select` did
not help either, because `select` survived and `select` is what it writes.

So stop asking. The year is not something to guess: each member is a cell in a table, and every
table in the manifest carries its `report_year`. `select(argmax, members=[cell(df1), cell(df2)],
keys=[...])` and `select(argmax, members=[literal(2021), literal(2022)], keys=[...])` rank by the
same keys and differ only in what they hand back, so rewriting the first into the second changes
the answer from an amount to the year of that amount -- which is the question.

This repairs a shape the model produces reliably. It does not invent a program the model did not
write, and it never touches one that is already answering in the right dimension.
"""

from __future__ import annotations

from vifinqa.programs.ir import ArgExtremumExpr, CellExpr, LiteralExpr, ScalarExpr, SelectExpr

# The operators whose answer is one of the members rather than a function of all of them. `min`
# and `max` return the extreme *value*, so a year question that used them was asking for something
# else and is left alone.
_PICKING = frozenset({"argmin", "argmax"})


def _years_for(members: tuple[ScalarExpr, ...], variable_years: dict[str, int]) -> list[int] | None:
    """The report year behind each member, or None if any member is not a resolvable cell.

    Every member has to be a plain cell whose table year is known. A member that is itself an
    expression may still evaluate to an amount from one year, but which year is then a guess, and
    a guess is the thing this module exists to avoid.
    """
    years: list[int] = []
    for member in members:
        if not isinstance(member, CellExpr):
            return None
        year = variable_years.get(member.variable)
        if year is None:
            return None
        years.append(year)
    # Distinct, or the answer is ambiguous: two members from the same year cannot be told apart by
    # the thing we would be answering with.
    if len(set(years)) != len(years):
        return None
    return years


def retarget_year_answer(
    expression: ScalarExpr, *, target_unit: str, variable_years: dict[str, int]
) -> tuple[ScalarExpr, bool]:
    """Rewrite a ranked selection so it answers with a year, when a year is what was asked.

    Returns the expression and whether anything changed, so a caller can record that it fired
    rather than discovering the repair by comparing outputs.
    """
    if target_unit.upper() != "YEAR":
        return expression, False

    if isinstance(expression, SelectExpr):
        if expression.operator not in _PICKING:
            return expression, False
        # Conditions filter which members are eligible and are evaluated against the members'
        # own keys, so a filtered cohort still ranks correctly after the swap. What it must not
        # have is a mismatched key list -- that is a different fault with its own error.
        if expression.keys is not None and len(expression.keys) != len(expression.members):
            return expression, False
        years = _years_for(expression.members, variable_years)
        if years is None:
            return expression, False
        # Without an explicit key list a select ranks the members themselves, so the amounts have
        # to move into `keys` before the members become years -- otherwise it would rank years.
        keys = expression.keys if expression.keys is not None else expression.members
        return (
            SelectExpr(
                operator=expression.operator,
                members=tuple(LiteralExpr(value=float(year), dimension="YEAR") for year in years),
                conditions=expression.conditions,
                keys=tuple(keys),
            ),
            True,
        )

    if isinstance(expression, ArgExtremumExpr):
        years = _years_for(expression.values, variable_years)
        if years is None or len(expression.keys) != len(expression.values):
            return expression, False
        return (
            ArgExtremumExpr(
                mode=expression.mode,
                keys=expression.keys,
                values=tuple(LiteralExpr(value=float(year), dimension="YEAR") for year in years),
            ),
            True,
        )

    return expression, False
