from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

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
        model_revision: str | None = None,
        device: str | None = None,
        max_length: int = 8_192,
        use_fp16: bool = False,
    ) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self.model = CrossEncoder(
            model_id,
            revision=model_revision,
            device=device,
            max_length=max_length,
        )
        self.max_length = max_length
        if use_fp16 and device is not None and device.startswith("cuda"):
            self.model.model.half()

    @staticmethod
    def document_text(record: ManifestRecord) -> str:
        """Remove the duplicated ASCII retrieval view before cross-encoding.

        BGE-M3 sparse/dense retrieval benefits from the accentless expansion, but a
        multilingual cross-encoder should judge the original financial card once.
        This preserves every original label while nearly halving common inputs.
        """
        original, separator, _ascii_view = record.retrieval_text.partition("\nkhong dau:")
        return original if separator else record.retrieval_text

    def rerank(
        self,
        question: str,
        candidates: list[ManifestRecord],
        *,
        top_k: int = 10,
        batch_size: int = 16,
        max_batch_tokens: int | None = None,
    ) -> list[RerankedHit]:
        if not candidates or top_k <= 0:
            return []
        if batch_size <= 0 or (max_batch_tokens is not None and max_batch_tokens <= 0):
            raise ValueError("batch_size and max_batch_tokens must be positive")
        pairs = [(question, self.document_text(record)) for record in candidates]
        scores: list[float]
        if max_batch_tokens is None:
            scores = [
                float(score)
                for score in self.model.predict(
                    pairs,
                    batch_size=batch_size,
                    show_progress_bar=False,
                )
            ]
        else:
            tokenized: dict[str, Any] = self.model.tokenizer(
                [pair[0] for pair in pairs],
                [pair[1] for pair in pairs],
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_length=True,
            )
            raw_lengths = tokenized.get("length")
            if not isinstance(raw_lengths, list) or len(raw_lengths) != len(pairs):
                raise ValueError("Cross-encoder tokenizer did not return pair lengths")
            buckets: dict[int, list[int]] = {}
            for index, raw_length in enumerate(raw_lengths):
                length = int(raw_length)
                effective_batch = min(batch_size, max(1, max_batch_tokens // max(1, length)))
                buckets.setdefault(effective_batch, []).append(index)
            scores = [0.0] * len(pairs)
            for effective_batch, indices in sorted(buckets.items()):
                bucket_scores = self.model.predict(
                    [pairs[index] for index in indices],
                    batch_size=effective_batch,
                    show_progress_bar=False,
                )
                for index, score in zip(indices, bucket_scores, strict=True):
                    scores[index] = float(score)
        ordered = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: (-float(item[1]), item[0].table_ref),
        )
        return [
            RerankedHit(record.table_ref, float(score), rank)
            for rank, (record, score) in enumerate(ordered[:top_k], start=1)
        ]


def preserve_route_coverage(
    hits: Sequence[RerankedHit],
    records_by_ref: dict[str, ManifestRecord],
    *,
    routes: Sequence[tuple[str | None, int | None]],
    scopes: set[str] | None,
    top_k: int,
) -> list[RerankedHit]:
    """Keep the strongest candidate for every requested entity/period route."""
    if top_k <= 0:
        return []
    ordered = list(hits)
    required: set[str] = set()
    for ticker, year in routes:
        match = next(
            (
                hit
                for hit in ordered
                if (record := records_by_ref[hit.table_ref])
                and (ticker is None or record.ticker == ticker)
                and (year is None or record.report_year == year)
                and (not scopes or record.scope in scopes)
            ),
            None,
        )
        if match is not None:
            required.add(match.table_ref)
    selected_refs = set(required)
    for hit in ordered:
        if len(selected_refs) >= top_k:
            break
        selected_refs.add(hit.table_ref)
    selected = [hit for hit in ordered if hit.table_ref in selected_refs][:top_k]
    return [
        RerankedHit(hit.table_ref, hit.score, rank) for rank, hit in enumerate(selected, start=1)
    ]
