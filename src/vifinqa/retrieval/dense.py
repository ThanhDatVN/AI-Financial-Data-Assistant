from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer

from vifinqa.indexing.manifest import ManifestRecord
from vifinqa.parsing.normalize import ascii_words


@dataclass(frozen=True, slots=True)
class DenseHit:
    table_ref: str
    score: float
    rank: int
    ticker: str
    report_year: int
    scope: str


class DenseIndex:
    def __init__(
        self,
        index: faiss.Index,
        records: list[ManifestRecord],
        *,
        model_id: str,
        model_revision: str | None = None,
        model: SentenceTransformer | None = None,
    ) -> None:
        self.index = index
        self.records = records
        self.model_id = model_id
        self.model_revision = model_revision
        self._model = model
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

    def _encoder(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_id, revision=self.model_revision)
        return self._model

    @classmethod
    def build(
        cls,
        records: list[ManifestRecord],
        *,
        model_id: str = "BAAI/bge-m3",
        model_revision: str | None = None,
        batch_size: int = 16,
        device: str | None = None,
    ) -> DenseIndex:
        if not records:
            raise ValueError("Cannot build a dense index over an empty manifest")
        model = SentenceTransformer(model_id, revision=model_revision, device=device)
        texts = [record.retrieval_text for record in records]
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return cls(
            index,
            records,
            model_id=model_id,
            model_revision=model_revision,
            model=model,
        )

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
    ) -> list[DenseHit]:
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
        query_text = query + "\n" + ascii_words(query)
        vector = (
            self._encoder()
            .encode([query_text], normalize_embeddings=True, convert_to_numpy=True)
            .astype(np.float32)
        )
        population = len(allowed_indices) if allowed_indices is not None else len(self.records)
        k = min(max(candidate_k, top_k), population)
        if allowed_indices is None:
            scores, indices = self.index.search(vector, k)
        else:
            candidate_vectors = self.index.reconstruct_batch(allowed_indices)
            candidate_scores = candidate_vectors @ vector[0]
            order = np.argsort(-candidate_scores, kind="stable")[:k]
            scores = candidate_scores[order][None, :]
            indices = allowed_indices[order][None, :]
        hits: list[DenseHit] = []
        for raw_index, raw_score in zip(indices[0], scores[0], strict=True):
            if raw_index < 0:
                continue
            record = self.records[int(raw_index)]
            if tickers and record.ticker not in tickers:
                continue
            if years and record.report_year not in years:
                continue
            if allowed_scopes and record.scope not in allowed_scopes:
                continue
            hits.append(
                DenseHit(
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
        faiss.write_index(self.index, str(path / "index.faiss"))
        (path / "config.json").write_text(
            json.dumps(
                {
                    "model_id": self.model_id,
                    "model_revision": self.model_revision,
                    "tables": len(self.records),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        with (path / "records.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for record in self.records:
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: Path) -> DenseIndex:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        records: list[ManifestRecord] = []
        with (path / "records.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(ManifestRecord.from_dict(json.loads(line)))
        index = faiss.read_index(str(path / "index.faiss"))
        if index.ntotal != len(records):
            raise ValueError("Dense index and record manifest have different lengths")
        revision = config.get("model_revision")
        return cls(
            index,
            records,
            model_id=str(config["model_id"]),
            model_revision=str(revision) if revision is not None else None,
        )
