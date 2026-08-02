from __future__ import annotations

from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from vifinqa.indexing.manifest import ManifestRecord


@dataclass(frozen=True, slots=True)
class RerankedHit:
    table_ref: str
    score: float
    rank: int


class CrossEncoderReranker:
    def __init__(
        self,
        model_id: str = "BAAI/bge-reranker-v2-m3",
        *,
        device: str | None = None,
    ) -> None:
        self.model = CrossEncoder(model_id, device=device)

    def rerank(
        self,
        question: str,
        candidates: list[ManifestRecord],
        *,
        top_k: int = 10,
        batch_size: int = 16,
    ) -> list[RerankedHit]:
        if not candidates or top_k <= 0:
            return []
        pairs = [(question, record.retrieval_text) for record in candidates]
        scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        ordered = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: (-float(item[1]), item[0].table_ref),
        )
        return [
            RerankedHit(record.table_ref, float(score), rank)
            for rank, (record, score) in enumerate(ordered[:top_k], start=1)
        ]
