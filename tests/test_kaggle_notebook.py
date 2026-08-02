from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER_NOTEBOOK = ROOT / "notebooks/01_kaggle_build_dense_artifact.ipynb"
GENERATION_NOTEBOOK = ROOT / "notebooks/02_kaggle_dense_and_generate.ipynb"


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
    assert 'INPUT_ROOT.rglob("dense-checkpoints/bge_m3/config.json")' in code
    assert '"artifact_type": "vifinqa_bge_m3_dense_index"' in code
    assert '"--max-runtime-minutes"' in code
    assert '"artifact_type": "vifinqa_bge_m3_dense_checkpoint"' in code
    assert "completed_shards" in code


def test_kaggle_generation_notebook_is_valid_and_pinned() -> None:
    code = _compiled_code(GENERATION_NOTEBOOK)

    assert "Qwen/Qwen3-8B-AWQ" in code
    assert "4da05a8edb55c6046cce958586c33b61da07bb79" in code
    assert 'THINKING_MODE = os.environ.get("VIFINQA_THINKING_MODE"' in code
    assert "SMOKE_IDS = [1, 213, 399, 442, 473]" in code
    assert "--default-chat-template-kwargs" in code
    assert "--thinking-mode" in code
    assert code.count("--max-attempts") == 2
    assert 'dense_config.get("model_revision") == DENSE_REVISION' in code
    assert "scripts/22_build_dense.py" not in code
    assert "BAAI/bge-reranker-v2-m3" in code
    assert "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e" in code
    assert '"scripts/33_rerank_retrieval.py"' in code
    assert '"scripts/34_merge_retrieval_shards.py"' in code
    assert '"--candidate-tables", "100"' in code
    assert '"--max-length", "8192"' in code
    assert '"--max-batch-tokens", "8192"' in code
    assert "retrieval_rerank_shards/shard_0/run_metadata.json" in code
    assert '"--data-parallel-size"' in code
    assert '"--shard-count"' in code
    assert code.count('"--project-revision"') == 4
    assert '"scripts/51_merge_generation_shards.py"' in code
    assert 'INPUT_ROOT.rglob("generation_qwen3_8b_awq_shards/shard_0/run_metadata.json")' in code
    assert "VLLM_CONFIG" in code
    assert "Qwen2.5" not in code


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
