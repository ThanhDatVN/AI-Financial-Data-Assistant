from __future__ import annotations

import sys
from pathlib import Path

import pytest

from vifinqa.kaggle_inputs import describe_inputs, iter_input_paths


def _symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError) as error:  # Windows without developer mode
        pytest.skip(f"symlinks unavailable: {error}")


def _dataset(root: Path, name: str) -> Path:
    dataset = root / name / "ViFinQA"
    (dataset / "questions").mkdir(parents=True)
    (dataset / "financial_statements").mkdir()
    (dataset / "questions/questions.jsonl").write_text("{}\n", encoding="utf-8")
    (dataset / "code_stock.csv").write_text("ticker\n", encoding="utf-8")
    return dataset


def test_iter_input_paths_follows_symlinked_dataset_mounts(tmp_path: Path) -> None:
    storage = tmp_path / "storage"
    dataset = _dataset(storage, "vifinqa")
    mounts = tmp_path / "input"
    (mounts / "datasets").mkdir(parents=True)
    _symlink(mounts / "datasets/vifinqa", storage / "vifinqa")

    matches = iter_input_paths("questions/questions.jsonl", root=mounts)

    assert [path.resolve() for path in matches] == [
        (dataset / "questions/questions.jsonl").resolve()
    ]
    if sys.version_info >= (3, 13):
        # The regression this guards against: since 3.13 `**` expansion no longer follows
        # symlinks, so rglob silently finds nothing under Kaggle's symlinked mounts.
        assert not list(mounts.rglob("questions.jsonl"))


def test_iter_input_paths_is_sorted_deduplicated_and_depth_bounded(tmp_path: Path) -> None:
    mounts = tmp_path / "input"
    _dataset(mounts, "b-dataset")
    _dataset(mounts, "a-dataset")
    deep = mounts / "deep"
    _dataset(deep / "one/two/three", "nested")

    matches = iter_input_paths("questions/questions.jsonl", root=mounts)
    assert [path.relative_to(mounts).as_posix() for path in matches] == [
        "a-dataset/ViFinQA/questions/questions.jsonl",
        "b-dataset/ViFinQA/questions/questions.jsonl",
        "deep/one/two/three/nested/ViFinQA/questions/questions.jsonl",
    ]

    shallow = iter_input_paths("questions/questions.jsonl", root=mounts, max_depth=3)
    assert [path.relative_to(mounts).as_posix() for path in shallow] == [
        "a-dataset/ViFinQA/questions/questions.jsonl",
        "b-dataset/ViFinQA/questions/questions.jsonl",
    ]


def test_iter_input_paths_survives_symlink_cycles(tmp_path: Path) -> None:
    mounts = tmp_path / "input"
    _dataset(mounts, "vifinqa")
    _symlink(mounts / "vifinqa/loop", mounts)

    assert len(iter_input_paths("questions/questions.jsonl", root=mounts)) == 1


def test_iter_input_paths_returns_empty_without_mounts(tmp_path: Path) -> None:
    assert iter_input_paths("questions/questions.jsonl", root=tmp_path / "missing") == []


def test_describe_inputs_names_what_is_mounted(tmp_path: Path) -> None:
    mounts = tmp_path / "input"
    _dataset(mounts, "vifinqa")

    inventory = describe_inputs(root=mounts)

    assert "vifinqa/" in inventory
    assert "ViFinQA/" in inventory
    assert "code_stock.csv" in inventory
    assert describe_inputs(root=tmp_path / "missing").endswith("does not exist")
