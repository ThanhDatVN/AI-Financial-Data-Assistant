from __future__ import annotations

from collections.abc import Sequence


def balanced_round_robin(rankings: Sequence[Sequence[str]], *, limit: int) -> list[str]:
    if limit <= 0:
        return []
    output: list[str] = []
    seen: set[str] = set()
    depth = 0
    while len(output) < limit and any(depth < len(ranking) for ranking in rankings):
        for ranking in rankings:
            if depth < len(ranking) and ranking[depth] not in seen:
                output.append(ranking[depth])
                seen.add(ranking[depth])
                if len(output) >= limit:
                    break
        depth += 1
    return output


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], *, rrf_k: int = 60
) -> list[tuple[str, float]]:
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    order = 0
    for ranking in rankings:
        seen_in_ranking: set[str] = set()
        for rank, item in enumerate(ranking, start=1):
            if item in seen_in_ranking:
                continue
            seen_in_ranking.add(item)
            first_seen.setdefault(item, order)
            order += 1
            scores[item] = scores.get(item, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], first_seen[item[0]], item[0]))


def coverage_budget(top_k: int, route_count: int) -> int:
    if top_k <= 0 or route_count <= 0:
        raise ValueError("top_k and route_count must be positive")
    return max(top_k, route_count)
