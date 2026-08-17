"""Spend the prompt's candidate slots on the statement the question is actually asking about.

Only 37% of questions say whether they mean the consolidated or the separate statement. For the
rest the pipeline has been showing the model both, and packages `z2`/`z3` measured what that
costs: naming both scopes scored DOCS F2 0.8595 while defaulting to consolidated scored 0.9548,
which puts the default's accuracy near 93% ([docs/12 section 9.2](../../../docs/12-kinh-nghiem.md)).

The same default applied to the candidate list buys depth rather than points. Measured on the
ranking the scored run 3119 used, 27.5% of the top-20 is a table from the other scope; dropping
those pulls the twentieth surviving candidate down to original rank 30 on average. Nothing else
in the pipeline reaches deeper for free -- the context is full at twenty tables.

Two guards keep the rule from taking questions away:

* a table whose scope the manifest could not read is not evidence of the wrong scope, so
  `unknown` is never dropped;
* a ticker-year that has no table of the wanted scope keeps everything it has, because some
  companies only ever filed one of the two.

Together they left all 1,012 questions with candidates in the sizing run.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from vifinqa.indexing.manifest import ManifestRecord

# The scopes a document name states outright. `unknown` is the manifest saying it could not tell,
# which is a different thing from saying "separate", and has to be treated as such.
DECLARED_SCOPES = frozenset({"consolidated", "separate", "aggregated"})

# What an unstated question is taken to mean. Measured at roughly 93% correct, against a 83.3%
# break-even for naming both ([docs/12 section 8.3](../../../docs/12-kinh-nghiem.md)).
DEFAULT_SCOPE = "consolidated"

# The policy that leaves the ranking exactly as it arrived.
NO_ROUTING = "both"


def scope_routed(
    table_refs: Iterable[str],
    *,
    records: Mapping[str, ManifestRecord],
    scope: str | None,
    policy: str = NO_ROUTING,
) -> list[str]:
    """Return the ranking with the other statement's tables removed, order untouched.

    `scope` is what the question said, or None when it said nothing; `policy` is what to assume
    in that case. `policy=NO_ROUTING` disables the filter so a caller can measure against the
    ranking as it stands.
    """
    ranking = [str(ref) for ref in table_refs]
    if policy == NO_ROUTING:
        return ranking
    target = scope or policy
    if target not in DECLARED_SCOPES:
        return ranking
    available: dict[tuple[str, int], set[str]] = {}
    for ref in ranking:
        record = records.get(ref)
        if record is not None:
            available.setdefault((record.ticker, record.report_year), set()).add(record.scope)
    kept: list[str] = []
    for ref in ranking:
        record = records.get(ref)
        contradicts = (
            record is not None
            and record.scope in DECLARED_SCOPES
            and record.scope != target
            # The second guard: a ticker-year that filed only the other statement keeps it,
            # because dropping it would leave the question with nothing at all.
            and target in available[(record.ticker, record.report_year)]
        )
        if not contradicts:
            kept.append(ref)
    return kept
