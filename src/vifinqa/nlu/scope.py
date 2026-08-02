from __future__ import annotations

from dataclasses import dataclass

from vifinqa.parsing.normalize import ascii_compact


@dataclass(frozen=True, slots=True)
class ScopeIntent:
    scope: str | None
    explicit: bool


def detect_scope_intent(text: str) -> ScopeIntent:
    compact = ascii_compact(text)
    parent = any(marker in compact for marker in ("congtyme", "baocaorieng", "bctcrieng"))
    consolidated = any(marker in compact for marker in ("hopnhat", "baocaohopnhat", "bctchn"))
    if parent and not consolidated:
        return ScopeIntent("separate", True)
    if consolidated and not parent:
        return ScopeIntent("consolidated", True)
    return ScopeIntent(None, False)
