from __future__ import annotations

from importlib import import_module
from pathlib import Path

from vifinqa.retrieval.fusion import coverage_budget

ROOT = Path(__file__).resolve().parents[1]


def test_candidate_budget_never_truncates_metadata_routes() -> None:
    assert coverage_budget(20, 7) == 20
    assert coverage_budget(20, 30) == 30


def test_a_bm25_only_run_does_not_need_the_dense_stack_installed() -> None:
    """BM25 alone beat both rankings built on the dense index, so it must run without one.

    Measured on one scored run, one submission each: BM25 put the gold table in the prompt for
    0.6168 of questions, BM25 plus dense fusion for 0.5997, and both plus the cross-encoder for
    0.5642. A top-level `faiss` import would make the losing stage a prerequisite for skipping
    it, and this environment has no faiss at all, so the import is the test.
    """
    runner = import_module("scripts.30_retrieve_questions")
    assert not hasattr(runner, "DenseIndex")


def test_a_ranking_is_written_beside_its_own_provenance() -> None:
    """A final run asserts this sidecar exists before it starts, and nothing was writing it.

    Only the reranker's merge step ever produced `<ranking>.metadata.json`, so choosing the
    cheaper and better-scoring BM25 ranking would have failed that assertion after the model was
    already loaded -- and the resume notebook copies the same file, so a partial run could not
    have been finished either.
    """
    source = (ROOT / "scripts/30_retrieve_questions.py").read_text(encoding="utf-8")
    assert 'args.output.with_suffix(args.output.suffix + ".metadata.json")' in source
    assert '"retrieval_sha256": _sha256(args.output)' in source
    assert '"ranking": "bm25" if dense is None else "hybrid"' in source
