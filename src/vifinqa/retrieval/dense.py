from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

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
        device: str | None = None,
        max_seq_length: int | None = None,
        use_fp16: bool = False,
    ) -> None:
        self.index = index
        self.records = records
        self.model_id = model_id
        self.model_revision = model_revision
        self._model = model
        self.device = device
        self.max_seq_length = max_seq_length
        self.use_fp16 = use_fp16
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
            self._model = SentenceTransformer(
                self.model_id,
                revision=self.model_revision,
                device=self.device,
            )
            if self.max_seq_length is not None:
                self._model.max_seq_length = self.max_seq_length
            if self.use_fp16 and str(self._model.device).startswith("cuda"):
                self._model.half()
        return self._model

    @staticmethod
    def _corpus_sha256(records: list[ManifestRecord]) -> str:
        digest = hashlib.sha256()
        for record in records:
            for value in (record.table_ref, record.retrieval_text):
                encoded = value.encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
        return digest.hexdigest()

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _save_array_atomic(path: Path, values: npt.NDArray[np.float32]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
        temporary.replace(path)

    @staticmethod
    def _token_lengths(
        model: SentenceTransformer,
        texts: list[str],
        *,
        chunk_size: int = 512,
    ) -> list[int]:
        lengths: list[int] = []
        for start in range(0, len(texts), chunk_size):
            tokenized = model.tokenize(texts[start : start + chunk_size])
            attention_mask = tokenized.get("attention_mask")
            if attention_mask is None:
                raise ValueError("SentenceTransformer tokenizer did not return an attention mask")
            lengths.extend(int(value) for value in attention_mask.sum(dim=1).tolist())
        return lengths

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
            device=device,
        )

    @classmethod
    def build_checkpointed(
        cls,
        records: list[ManifestRecord],
        *,
        checkpoint_dir: Path,
        model_id: str = "BAAI/bge-m3",
        model_revision: str | None = None,
        batch_size: int = 16,
        checkpoint_size: int = 256,
        max_seq_length: int = 8_192,
        max_batch_tokens: int = 8_192,
        sort_by_length: bool = False,
        device: str | list[str] | None = None,
        use_fp16: bool = False,
        max_runtime_seconds: float | None = None,
    ) -> DenseIndex | None:
        if not records:
            raise ValueError("Cannot build a dense index over an empty manifest")
        if batch_size <= 0 or checkpoint_size <= 0 or max_seq_length <= 0 or max_batch_tokens <= 0:
            raise ValueError("Batch, checkpoint, sequence, and token-budget sizes must be positive")
        if max_runtime_seconds is not None and max_runtime_seconds <= 0:
            raise ValueError("max_runtime_seconds must be positive")
        devices = [device] if isinstance(device, str) else (device or [])
        model_device = "cpu" if len(devices) > 1 else (devices[0] if devices else None)
        model = SentenceTransformer(model_id, revision=model_revision, device=model_device)
        model.max_seq_length = max_seq_length
        if use_fp16 and any(value.startswith("cuda") for value in devices):
            model.half()
        dimension = model.get_sentence_embedding_dimension()
        if dimension is None:
            raise ValueError(f"Model {model_id} did not report an embedding dimension")
        token_lengths = cls._token_lengths(model, [record.retrieval_text for record in records])
        if sort_by_length:
            ordered = sorted(
                zip(token_lengths, records, strict=True),
                key=lambda item: (item[0], item[1].table_ref),
                reverse=True,
            )
            token_lengths = [length for length, _record in ordered]
            records = [record for _length, record in ordered]
        expected_config = {
            "format_version": 1,
            "model_id": model_id,
            "model_revision": model_revision,
            "max_seq_length": max_seq_length,
            "max_batch_tokens": max_batch_tokens,
            "use_fp16": use_fp16,
            "dimension": dimension,
            "tables": len(records),
            "corpus_sha256": cls._corpus_sha256(records),
            "runtime_versions": {
                distribution: version(distribution)
                for distribution in ("sentence-transformers", "torch", "transformers")
            },
        }
        checkpoint_config = checkpoint_dir / "config.json"
        if checkpoint_config.exists():
            actual_config = json.loads(checkpoint_config.read_text(encoding="utf-8"))
            if actual_config != expected_config:
                raise ValueError(
                    "Dense checkpoint settings or corpus changed; use a new checkpoint directory."
                )
        else:
            cls._write_json_atomic(checkpoint_config, expected_config)

        shards = [
            (
                start,
                min(start + checkpoint_size, len(records)),
                checkpoint_dir
                / (
                    f"embeddings_{start:06d}_"
                    f"{min(start + checkpoint_size, len(records)):06d}.npy"
                ),
            )
            for start in range(0, len(records), checkpoint_size)
        ]
        missing_rows = sum(stop - start for start, stop, path in shards if not path.exists())
        missing_tokens = sum(
            sum(token_lengths[start:stop]) for start, stop, path in shards if not path.exists()
        )
        print(
            f"dense checkpoints: {len(shards)} shards, "
            f"{len(records) - missing_rows}/{len(records)} rows already persisted"
        )
        started_at = time.monotonic()
        encoded_rows = 0
        encoded_tokens = 0
        pool: dict[Literal["input", "output", "processes"], Any] | None = None
        if len(devices) > 1 and missing_rows:
            pool = model.start_multi_process_pool(target_devices=devices)
        try:
            for start, stop, shard_path in shards:
                expected_shape = (stop - start, dimension)
                if shard_path.exists():
                    existing = np.load(shard_path, mmap_mode="r", allow_pickle=False)
                    if existing.shape != expected_shape or existing.dtype != np.float32:
                        raise ValueError(f"Invalid dense checkpoint shard: {shard_path}")
                    print(f"reusing dense checkpoint {shard_path.name}")
                    continue
                if (
                    max_runtime_seconds is not None
                    and encoded_rows
                    and time.monotonic() - started_at >= max_runtime_seconds
                ):
                    print(
                        "dense time budget reached after an atomic shard; "
                        "save the checkpoint directory and resume in another session"
                    )
                    break
                texts = [record.retrieval_text for record in records[start:stop]]
                longest = max(token_lengths[start:stop])
                effective_batch_size = min(batch_size, max(1, max_batch_tokens // longest))
                print(
                    f"encoding rows {start}:{stop}; longest={longest} tokens, "
                    f"batch_size={effective_batch_size}"
                )
                encoded = model.encode(
                    texts,
                    batch_size=effective_batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=True,
                    pool=pool,
                ).astype(np.float32)
                if encoded.shape != expected_shape or not np.isfinite(encoded).all():
                    raise ValueError(f"Invalid embeddings for rows {start}:{stop}: {encoded.shape}")
                cls._save_array_atomic(shard_path, encoded)
                encoded_rows += stop - start
                encoded_tokens += sum(token_lengths[start:stop])
                elapsed = max(time.monotonic() - started_at, 1e-9)
                rows_per_second = encoded_rows / elapsed
                tokens_per_second = encoded_tokens / elapsed
                remaining_tokens = missing_tokens - encoded_tokens
                eta_hours = remaining_tokens / tokens_per_second / 3_600
                print(
                    f"saved dense checkpoint {shard_path.name} ({stop}/{len(records)}); "
                    f"{rows_per_second:.1f} rows/s, {tokens_per_second:.0f} tokens/s, "
                    f"token-weighted ETA {eta_hours:.2f} h"
                )
        finally:
            if pool is not None:
                model.stop_multi_process_pool(pool)

        remaining_shards = [path for _start, _stop, path in shards if not path.exists()]
        if remaining_shards:
            print(
                f"dense build paused safely: {len(shards) - len(remaining_shards)}/"
                f"{len(shards)} shards persisted"
            )
            return None

        index = faiss.IndexFlatIP(dimension)
        for _start, _stop, shard_path in shards:
            shard = np.load(shard_path, mmap_mode="r", allow_pickle=False)
            index.add(np.asarray(shard, dtype=np.float32))
        if index.ntotal != len(records):
            raise ValueError(
                f"Dense checkpoint produced {index.ntotal} vectors for {len(records)} rows"
            )
        primary_device = devices[0] if devices else None
        return cls(
            index,
            records,
            model_id=model_id,
            model_revision=model_revision,
            model=model if len(devices) <= 1 else None,
            device=primary_device,
            max_seq_length=max_seq_length,
            use_fp16=use_fp16,
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
                    "max_seq_length": self.max_seq_length,
                    "use_fp16": self.use_fp16,
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
    def load(cls, path: Path, *, device: str | None = None) -> DenseIndex:
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
            device=device,
            max_seq_length=(
                int(config["max_seq_length"]) if config.get("max_seq_length") is not None else None
            ),
            use_fp16=bool(config.get("use_fp16", False)),
        )
