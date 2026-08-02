from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/02_kaggle_dense_and_generate.ipynb"


def test_kaggle_generation_notebook_is_valid_and_pinned() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{NOTEBOOK}:cell-{index}", "exec")

    assert "Qwen/Qwen3-8B-AWQ" in code
    assert "4da05a8edb55c6046cce958586c33b61da07bb79" in code
    assert 'THINKING_MODE = os.environ.get("VIFINQA_THINKING_MODE"' in code
    assert "SMOKE_IDS = [1, 213, 399, 442, 473]" in code
    assert "--default-chat-template-kwargs" in code
    assert "--thinking-mode" in code
    assert code.count("--max-attempts") == 2
    assert 'dense_config.get("model_revision") == DENSE_REVISION' in code
    assert "Qwen2.5" not in code


def test_kaggle_runtime_pins_are_dependency_compatible() -> None:
    requirements = (ROOT / "requirements-gpu.txt").read_text(encoding="utf-8")
    assert "vllm==0.19.1" in requirements
    assert "sentence-transformers==5.5.1" in requirements
    assert "openai>=2,<3" in requirements
    assert "openai>=1" not in requirements
