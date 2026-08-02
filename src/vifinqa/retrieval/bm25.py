from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import bm25s
import numpy as np
import numpy.typing as npt

from vifinqa.indexing.manifest import ManifestRecord
from vifinqa.parsing.normalize import ascii_words


@dataclass(frozen=True, slots=True)
class BM25Hit:
    table_ref: str
    score: float
    rank: int
    ticker: str
    report_year: int
    scope: str


@dataclass(frozen=True, slots=True)
class BM25Document:
    table_ref: str
    ticker: str
    report_year: int
    scope: str


def _search_text(text: str) -> str:
    return text + "\n" + ascii_words(text)


class BM25Index:
    def __init__(self, retriever: bm25s.BM25, records: list[BM25Document]) -> None:
        self.retriever = retriever
        self.records = records
        groups: dict[tuple[str, int, str], list[int]] = {}
        for index, record in enumerate(records):
            groups.setdefault((record.ticker, record.report_year, record.scope), []).append(index)
        self._metadata_groups = groups

    def _allowed_indices(
        self,
        *,
        tickers: set[str] | None,
        years: set[int] | None,
        scopes: set[str] | None,
    ) -> npt.NDArray[np.int64] | None:
        if tickers is None and years is None and scopes is None:
            return None
        indices = [
            index
            for (ticker, year, scope), group in self._metadata_groups.items()
            if (not tickers or ticker in tickers)
            and (not years or year in years)
            and (not scopes or scope in scopes)
            for index in group
        ]
        return np.asarray(indices, dtype=np.int64)

    @classmethod
    def build(cls, records: list[ManifestRecord]) -> BM25Index:
        if not records:
            raise ValueError("Cannot build BM25 over an empty manifest")
        corpus = [record.retrieval_text for record in records]
        tokens = bm25s.tokenize(corpus, stopwords=None, show_progress=False)
        retriever = bm25s.BM25(method="lucene")
        retriever.index(tokens, show_progress=False)
        documents = [
            BM25Document(record.table_ref, record.ticker, record.report_year, record.scope)
            for record in records
        ]
        return cls(retriever, documents)

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        candidate_k: int = 1_000,
        tickers: set[str] | None = None,
        years: set[int] | None = None,
        scope: str | None = None,
        scopes: set[str] | None = None,
    ) -> list[BM25Hit]:
        if top_k <= 0:
            return []
        if scope is not None and scopes is not None:
            raise ValueError("Pass scope or scopes, not both")
        allowed_scopes = scopes if scopes is not None else ({scope} if scope else None)
        allowed_indices = self._allowed_indices(
            tickers=tickers,
            years=years,
            scopes=allowed_scopes,
        )
        if allowed_indices is not None and not len(allowed_indices):
            return []
        metadata_terms = " ".join(
            [
                *(sorted(tickers) if tickers else []),
                *(str(year) for year in sorted(years or [])),
                *(sorted(allowed_scopes) if allowed_scopes else []),
            ]
        )
        query_tokens = bm25s.tokenize(
            [_search_text(query + "\n" + metadata_terms)], stopwords=None, show_progress=False
        )
        population = len(allowed_indices) if allowed_indices is not None else len(self.records)
        k = min(max(candidate_k, top_k), population)
        weight_mask = None
        if allowed_indices is not None:
            weight_mask = np.zeros(len(self.records), dtype=np.float32)
            weight_mask[allowed_indices] = 1.0
        results = self.retriever.retrieve(
            query_tokens,
            k=k,
            show_progress=False,
            weight_mask=weight_mask,
        )
        hits: list[BM25Hit] = []
        for raw_index, raw_score in zip(results.documents[0], results.scores[0], strict=True):
            record = self.records[int(raw_index)]
            if tickers and record.ticker not in tickers:
                continue
            if years and record.report_year not in years:
                continue
            if allowed_scopes and record.scope not in allowed_scopes:
                continue
            hits.append(
                BM25Hit(
                    table_ref=record.table_ref,
                    score=float(raw_score),
                    rank=len(hits) + 1,
                    ticker=record.ticker,
                    report_year=record.report_year,
                    scope=record.scope,
                )
            )
            if len(hits) >= top_k:
                break
        return hits

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.retriever.save(path)
        with (path / "records.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for record in self.records:
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: Path, *, mmap: bool = True) -> BM25Index:
        retriever = bm25s.BM25.load(path, mmap=mmap)
        records: list[BM25Document] = []
        with (path / "records.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(BM25Document(**json.loads(line)))
        return cls(retriever, records)
