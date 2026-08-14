from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_dense_backend_does_not_require_bm25_dependency() -> None:
    # The point of this test is that the dense backend stays independent of bm25s, which can
    # only be shown where the dense backend imports at all. faiss lives in the Linux/GPU
    # requirements, so on a CPU checkout there is nothing to assert rather than a failure.
    pytest.importorskip("faiss")
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    probe = (
        "import sys; "
        "sys.modules['bm25s'] = None; "
        "from vifinqa.retrieval.dense import DenseIndex; "
        "assert DenseIndex.__name__ == 'DenseIndex'"
    )
    subprocess.run(
        [sys.executable, "-c", probe],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
