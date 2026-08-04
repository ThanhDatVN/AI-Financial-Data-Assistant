from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER_NOTEBOOK = ROOT / "notebooks/01_kaggle_build_dense_artifact.ipynb"
GENERATION_NOTEBOOK = ROOT / "notebooks/02_kaggle_dense_and_generate.ipynb"
KAGGLE_INPUTS = ROOT / "src/vifinqa/kaggle_inputs.py"


def _compiled_code(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path}:cell-{index}", "exec")
    return code


def test_kaggle_dense_builder_is_full_quality_resumable_and_multi_gpu() -> None:
    code = _compiled_code(BUILDER_NOTEBOOK)
    assert 'MODEL = "BAAI/bge-m3"' in code
    assert "5617a9f61b028005a4858fdac845db406aefb181" in code
    assert '"--max-seq-length",\n        "8192"' in code
    assert '"--max-batch-tokens",\n        "8192"' in code
    assert '"--sort-by-length"' in code
    assert '"--fp16"' not in code
    assert "torch.cuda.device_count() >= 2" in code
    assert '"--checkpoint-dir"' in code
    assert 'iter_input_paths("dense-checkpoints/bge_m3/config.json")' in code
    assert '"artifact_type": "vifinqa_bge_m3_dense_index"' in code
    assert '"--max-runtime-minutes"' in code
    assert '"artifact_type": "vifinqa_bge_m3_dense_checkpoint"' in code
    assert "completed_shards" in code


def test_kaggle_generation_notebook_is_valid_and_pinned() -> None:
    code = _compiled_code(GENERATION_NOTEBOOK)

    assert "Qwen/Qwen3-8B-AWQ" in code
    assert "4da05a8edb55c6046cce958586c33b61da07bb79" in code
    assert 'THINKING_MODE = os.environ.get("VIFINQA_THINKING_MODE"' in code
    assert "torch.cuda.device_count() >= requested_dp" in code
    assert "assert capability >= (" in code
    assert "select T4 x2" in code
    assert '"pull", "--ff-only", "origin", "main"' in code
    assert 'if (PROJECT / ".git").exists()' in code
    assert "SMOKE ERRORS (full unresolved records)" in code
    assert "generation_qwen3_8b_awq_smoke_{PROJECT_SHA[:12]}" in code
    assert "SMOKE_IDS = [1, 213, 399, 442, 473]" in code
    assert "WIDE GATE ERRORS (full unresolved records)" in code
    assert "def route_fan_out(" in code
    assert (
        'wide_ids = sorted(int(row["id"]) for row in retrieval_rows if route_fan_out(row) > 10)'
        in code
    )
    assert "1012 / (DP * wide_rate) / 3_600" in code
    assert "--default-chat-template-kwargs" in code
    assert "--thinking-mode" in code
    # smoke, widest-route gate, full run
    assert code.count("--max-attempts") == 3
    assert 'dense_config.get("model_revision") == DENSE_REVISION' in code
    assert "scripts/22_build_dense.py" not in code
    assert "BAAI/bge-reranker-v2-m3" in code
    assert "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e" in code
    assert '"scripts/33_rerank_retrieval.py"' in code
    assert '"scripts/34_merge_retrieval_shards.py"' in code
    assert '"--candidate-tables"' in code and '"100"' in code
    assert '"--max-length"' in code and '"8192"' in code
    assert '"--max-batch-tokens"' in code and '"8192"' in code
    assert "retrieval_rerank_shards/shard_0/run_metadata.json" in code
    assert "checkpoint_project_revision" in code
    assert '"hybrid_project_revision": HYBRID_PROJECT_SHA' in code
    assert '"reranker_project_revision": RERANK_PROJECT_SHA' in code
    assert '"--data-parallel-size"' in code
    assert '"--shard-count"' in code
    # hybrid retrieval, reranking, smoke, widest-route gate, full run
    assert code.count('"--project-revision"') == 5
    assert '"scripts/51_merge_generation_shards.py"' in code
    assert 'iter_input_paths("generation_qwen3_8b_awq_shards/shard_0/run_metadata.json")' in code
    assert "VLLM_CONFIG" in code
    assert "cuda_driver_linker_environment" in code
    assert 'linker_name = linker_dir / "libcuda.so"' in code
    assert 'for variable in ("LIBRARY_PATH", "LD_LIBRARY_PATH")' in code
    assert "env=server_environment" in code
    assert "[-50_000:]" in code
    assert "Qwen2.5" not in code


def test_notebooks_discover_inputs_through_symlinked_kaggle_mounts() -> None:
    module_source = KAGGLE_INPUTS.read_text(encoding="utf-8")
    helpers = module_source[module_source.index("def iter_input_paths") :].rstrip("\n")
    for notebook in (BUILDER_NOTEBOOK, GENERATION_NOTEBOOK):
        code = _compiled_code(notebook)
        # The bootstrap cell runs before the project is installed, so it carries a verbatim
        # copy of the module instead of importing it.
        assert helpers in code, f"{notebook.name} must copy src/vifinqa/kaggle_inputs.py verbatim"
        assert "rglob" not in code, f"{notebook.name} must not rglob symlinked Kaggle mounts"
        assert "describe_inputs()" in code
        assert "INPUT_INVENTORY" in code


def test_kaggle_runtime_pins_are_dependency_compatible() -> None:
    requirements = (ROOT / "requirements-gpu.txt").read_text(encoding="utf-8")
    assert "vllm==0.19.1" in requirements
    assert "sentence-transformers==5.5.1" in requirements
    assert "transformers==5.5.3" in requirements
    assert "openai>=2,<3" in requirements
    assert "openai>=1" not in requirements
    dense_requirements = (ROOT / "requirements-dense.txt").read_text(encoding="utf-8")
    assert "vllm" not in dense_requirements
    assert "sentence-transformers==5.5.1" in dense_requirements
    assert "transformers==5.5.3" in dense_requirements
    assert "faiss-cpu==1.9.0.post1" in dense_requirements
    assert "Unidecode==1.3.8" in dense_requirements
    assert "ftfy==6.3.1" in dense_requirements


def test_dense_notebook_bootstraps_and_probes_backend_import() -> None:
    code = _compiled_code(BUILDER_NOTEBOOK)
    assert '"bm25s==0.2.6"' not in code
    assert '"pull", "--ff-only", "origin", "main"' in code
    assert "from vifinqa.retrieval.dense import DenseIndex" in code
    assert "DenseIndex.__name__ == 'DenseIndex'" in code
