"""Sparse, dense, fusion, and reranking components.

Retrieval backends have different optional dependencies.  Keep backend imports
lazy so a dense-only runtime does not need to install the BM25 stack.
"""

from typing import TYPE_CHECKING

from vifinqa.retrieval.fusion import reciprocal_rank_fusion

if TYPE_CHECKING:
    from vifinqa.retrieval.bm25 import BM25Hit, BM25Index

__all__ = ["BM25Hit", "BM25Index", "reciprocal_rank_fusion"]


def __getattr__(name: str) -> object:
    if name in {"BM25Hit", "BM25Index"}:
        from vifinqa.retrieval.bm25 import BM25Hit, BM25Index

        value = {"BM25Hit": BM25Hit, "BM25Index": BM25Index}[name]
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
