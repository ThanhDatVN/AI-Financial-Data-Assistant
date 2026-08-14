from __future__ import annotations

import copy
import re
from collections.abc import Iterable
from typing import Any

SEMANTIC_CONVENTION_VERSION = 1

# The organiser's reference prompt defines an undirected difference as a non-negative absolute
# value.  Keep this deliberately narrower than generic "change" language: questions that ask for
# an increase, decrease, or growth retain their direction and sign.
_UNDIRECTED_DIFFERENCE_RE = re.compile(
    r"\b(?:chênh\s+lệch|khác\s+biệt|cách\s+biệt|chênh\s+nhau)\b",
    flags=re.IGNORECASE,
)


def asks_for_undirected_difference(question: str) -> bool:
    """Whether the Vietnamese question explicitly asks for an unsigned difference."""
    return _UNDIRECTED_DIFFERENCE_RE.search(question) is not None


def normalize_absolute_difference(
    *, question: str, pandas_query: str, answer: float
) -> tuple[str, float, bool]:
    """Apply the organiser's unsigned-difference convention to one executed result."""
    if answer < 0 and asks_for_undirected_difference(question):
        return f"abs({pandas_query})", abs(answer), True
    return pandas_query, answer, False


def apply_absolute_difference_convention(
    rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[int]]:
    """Make negative undirected differences non-negative without breaking execution parity.

    Both the declared answer and the executable query must change together.  Only negative rows
    are touched, which makes the transformation idempotent and leaves already-correct positive
    differences byte-for-byte unchanged.
    """
    transformed = copy.deepcopy(list(rows))
    changed_ids: list[int] = []
    for row in transformed:
        question = row.get("question")
        answer = row.get("answer")
        query = row.get("pandas_query")
        if (
            isinstance(question, str)
            and isinstance(answer, int | float)
            and not isinstance(answer, bool)
            and isinstance(query, str)
            and query.strip()
        ):
            normalized_query, normalized_answer, changed = normalize_absolute_difference(
                question=question,
                pandas_query=query,
                answer=float(answer),
            )
            if changed:
                row["answer"] = normalized_answer
                row["pandas_query"] = normalized_query
                changed_ids.append(int(row["id"]))
    return transformed, changed_ids
