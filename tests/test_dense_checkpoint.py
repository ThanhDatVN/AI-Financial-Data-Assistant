from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from vifinqa.indexing.manifest import ManifestRecord
from vifinqa.retrieval import dense as dense_module
from vifinqa.retrieval.dense import DenseIndex


class FakeSentenceTransformer:
    encode_calls: list[list[str]] = []
    encode_batch_sizes: list[int] = []

    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        device: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.device = device or "cpu"
        self.max_seq_length = 8_192
        self.half_called = False

    def half(self) -> FakeSentenceTransformer:
        self.half_called = True
        return self

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        lengths = [min(len(text.split()), self.max_seq_length) for text in texts]
        width = max(lengths)
        attention_mask = torch.zeros((len(texts), width), dtype=torch.int64)
        for index, length in enumerate(lengths):
            attention_mask[index, :length] = 1
        return {"attention_mask": attention_mask}

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        self.encode_calls.append(list(texts))
        self.encode_batch_sizes.append(int(kwargs["batch_size"]))
        values = np.asarray(
            [[len(text), sum(map(ord, text)) % 101, index + 1] for index, text in enumerate(texts)],
            dtype=np.float32,
        )
        return values / np.linalg.norm(values, axis=1, keepdims=True)


def _record(index: int) -> ManifestRecord:
    return ManifestRecord(
        table_ref=f"AAA_2024|table_{index}",
        doc_id="AAA_2024",
        ticker="AAA",
        report_year=2024,
        scope="consolidated",
        table_id=index,
        page_no=index,
        line_no=index,
        char_offset=index,
        section_title="Section",
        unit="VND",
        header_rows=1,
        n_rows=2,
        n_cols=2,
        headers=("Metric", "2024"),
        row_labels=(f"Metric {index}",),
        retrieval_text=f"retrieval text {index}",
        source_path="AAA/report.html",
        html_sha256="0" * 64,
    )


def test_checkpointed_dense_build_resumes_completed_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dense_module, "SentenceTransformer", FakeSentenceTransformer)
    FakeSentenceTransformer.encode_calls.clear()
    FakeSentenceTransformer.encode_batch_sizes.clear()
    records = [_record(index) for index in range(5)]
    checkpoints = tmp_path / "checkpoints"

    first = DenseIndex.build_checkpointed(
        records,
        checkpoint_dir=checkpoints,
        model_id="fake/model",
        model_revision="a" * 40,
        batch_size=2,
        checkpoint_size=2,
        max_seq_length=2_048,
        device="cuda:1",
        use_fp16=True,
    )

    assert first.index.ntotal == 5
    assert [len(batch) for batch in FakeSentenceTransformer.encode_calls] == [2, 2, 1]
    assert len(list(checkpoints.glob("embeddings_*.npy"))) == 3
    config = json.loads((checkpoints / "config.json").read_text(encoding="utf-8"))
    assert config["max_seq_length"] == 2_048
    assert config["use_fp16"] is True
    assert set(config["runtime_versions"]) == {"sentence-transformers", "torch", "transformers"}

    FakeSentenceTransformer.encode_calls.clear()
    FakeSentenceTransformer.encode_batch_sizes.clear()
    resumed = DenseIndex.build_checkpointed(
        records,
        checkpoint_dir=checkpoints,
        model_id="fake/model",
        model_revision="a" * 40,
        batch_size=1,
        checkpoint_size=2,
        max_seq_length=2_048,
        device=["cuda:0", "cuda:1"],
        use_fp16=True,
    )

    assert resumed.index.ntotal == 5
    assert FakeSentenceTransformer.encode_calls == []


def test_checkpointed_dense_build_sorts_exact_lengths_and_adapts_batch_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dense_module, "SentenceTransformer", FakeSentenceTransformer)
    FakeSentenceTransformer.encode_calls.clear()
    FakeSentenceTransformer.encode_batch_sizes.clear()
    records = [_record(index) for index in range(3)]
    records[0] = dataclasses.replace(records[0], retrieval_text="short")
    records[1] = dataclasses.replace(records[1], retrieval_text="one two three four")
    records[2] = dataclasses.replace(records[2], retrieval_text="one two")

    built = DenseIndex.build_checkpointed(
        records,
        checkpoint_dir=tmp_path / "checkpoints",
        model_id="fake/model",
        batch_size=8,
        checkpoint_size=2,
        max_batch_tokens=4,
        sort_by_length=True,
    )

    assert [record.retrieval_text for record in built.records] == [
        "one two three four",
        "one two",
        "short",
    ]
    assert [batch[0] for batch in FakeSentenceTransformer.encode_calls] == [
        "one two three four",
        "short",
    ]
    assert FakeSentenceTransformer.encode_batch_sizes == [1, 4]


def test_checkpointed_dense_build_rejects_changed_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dense_module, "SentenceTransformer", FakeSentenceTransformer)
    records = [_record(1)]
    checkpoints = tmp_path / "checkpoints"
    DenseIndex.build_checkpointed(
        records,
        checkpoint_dir=checkpoints,
        model_id="fake/model",
        checkpoint_size=1,
        max_seq_length=2_048,
    )

    with pytest.raises(ValueError, match="settings or corpus changed"):
        DenseIndex.build_checkpointed(
            records,
            checkpoint_dir=checkpoints,
            model_id="fake/model",
            checkpoint_size=1,
            max_seq_length=1_024,
        )
