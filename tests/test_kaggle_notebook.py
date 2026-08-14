from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER_NOTEBOOK = ROOT / "notebooks/01_kaggle_build_dense_artifact.ipynb"
GENERATION_NOTEBOOK = ROOT / "notebooks/02_kaggle_dense_and_generate.ipynb"
RESUME_NOTEBOOK = ROOT / "notebooks/03_kaggle_resume_and_submit.ipynb"
PACKAGE_NOTEBOOK = ROOT / "notebooks/04_kaggle_package_submission.ipynb"
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

    assert "Qwen/Qwen3-14B-AWQ" in code
    assert "31c69efc29464b6bb0aee1398b5a7b50a99340c3" in code
    assert "Qwen/Qwen3-8B-AWQ" in code
    assert "4da05a8edb55c6046cce958586c33b61da07bb79" in code
    assert 'os.environ["VIFINQA_MODEL_PROFILE"]' in code
    assert 'os.environ["VIFINQA_EXPECTED_PROJECT_SHA"]' in code
    assert "Final 14B-profile run requires written organiser confirmation." in code
    assert 'THINKING_MODE = os.environ.get("VIFINQA_THINKING_MODE"' in code
    assert "torch.cuda.device_count() >= requested_dp" in code
    assert "assert capability >= (" in code
    assert "select T4 x2" in code
    assert '"fetch", "--depth", "1", "origin", GIT_REF' in code
    assert '"checkout", "--detach", "FETCH_HEAD"' in code
    assert 'os.environ["VIFINQA_GIT_REF"]' in code
    assert 'if (PROJECT / ".git").exists()' in code
    assert "SMOKE ERRORS (full unresolved records)" in code
    assert "generation_{MODEL_RUN_TAG}_smoke_{PROJECT_SHA[:12]}" in code
    assert "SMOKE_IDS = [1, 213, 399, 442, 473]" in code
    assert "WIDE GATE ERRORS (full unresolved records)" in code
    assert "def route_fan_out(" in code
    assert (
        'wide_ids = sorted(int(row["id"]) for row in retrieval_rows if route_fan_out(row) > 10)'
        in code
    )
    assert "1012 / (DP * wide_rate) / 3_600" in code
    # The smoke and widest-route sets are both the hard tail; only a seeded random draw can
    # say what the run scores or how long it takes.
    assert "random.Random(20260802).sample" in code
    assert "decode rate per request: about" in code
    assert "scripts/72_benchmark_structured_decoding.py" in code
    assert "structured_decoding_benchmark_{MODEL_RUN_TAG}_{PROJECT_SHA[:12]}.json" in code
    assert '"--model-revision"' in code
    assert '"--project-revision"' in code
    assert '"model_total_parameters_billions": MODEL_TOTAL_PARAMS_B' in code
    assert '"model_non_embedding_parameters_billions": MODEL_NON_EMBEDDING_PARAMS_B' in code
    assert "vllm_kernel_diagnostic_{MODEL_RUN_TAG}_{PROJECT_SHA[:12]}.log" in code
    assert "VIFINQA_RUN_SUBSET_200" in code
    assert 'os.environ["VIFINQA_UNIT_VARIANTS"] = "manifest"' in code
    assert 'os.environ["VIFINQA_TABLE_UNIT_SOURCE"] = "latest"' in code
    assert "imported unit checkpoint:" in code
    assert 'if set(unit_ablation) == {"manifest", "latest"}' in code
    assert "configs/experiment_subset_200.json" in code
    assert '("manifest", "latest")' in code
    assert "--table-unit-source" in code
    assert "comparison_manifest.json" in code
    assert "scripts/54_compare_generation_variants.py" in code
    assert "offline_comparison.json" in code
    assert "subset_config_sha256" in code
    assert "1012 * per_question / 3600" in code
    # The sample has to shard exactly like the full run or its projection is fiction.
    assert "SHARDS = DP * int(os.environ.get(" in code
    # Four sequences per replica is what the server accepts, so four client shards per
    # replica fill the batch; two left it half empty and doubled the projected run.
    assert 'os.environ["VIFINQA_SHARDS_PER_REPLICA"] = "4"' in code
    # A Kaggle session is shorter than the run, so an unfinished one has to say so.
    assert "resuming with {completed_rows()}/1012 questions already answered" in code
    assert "elif answered < 1012:" in code
    # A cancelled version cannot be attached as an input to the notebook that finishes the
    # run, so a session that will not reach 1,012 in time has to end on purpose instead.
    assert 'os.environ["VIFINQA_QUESTION_LIMIT"]' in code
    assert 'common_cmd += ["--limit", QUESTION_LIMIT]' in code
    assert "partial run finished cleanly" in code
    assert code.count("str(SHARDS)") == 3
    # Decode measured 3.7 tokens/s with one sequence per replica; batching is the lever.
    # Qwen3-14B-AWQ leaves room for roughly two concurrent sequences on a T4, not four, so the
    # width became a knob rather than a constant. The pin that matters is the default.
    assert '"--max-num-seqs",\n        str(MAX_NUM_SEQS),' in code
    assert 'MAX_NUM_SEQS = int(os.environ.get("VIFINQA_MAX_NUM_SEQS", "2"))' in code
    assert '"max_num_seqs": MAX_NUM_SEQS' in code
    assert "--default-chat-template-kwargs" in code
    assert "--thinking-mode" in code
    # smoke, widest-route gate, representative sample, unit ablation, full run
    assert code.count("--max-attempts") == 5
    assert 'dense_config.get("model_revision") == DENSE_REVISION' in code
    assert "scripts/22_build_dense.py" not in code
    assert "BAAI/bge-reranker-v2-m3" in code
    assert "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e" in code
    assert '"scripts/33_rerank_retrieval.py"' in code
    assert '"scripts/34_merge_retrieval_shards.py"' in code
    # Reranking is hours of cross-encoding whose value against labels is unmeasured, so the
    # hybrid ranking has to be usable on its own.
    assert "retrieval_hybrid_qc.json" in code
    assert "RETRIEVAL = HYBRID" in code
    rerank_cell = next(
        "".join(cell["source"])
        for cell in json.loads(GENERATION_NOTEBOOK.read_text(encoding="utf-8"))["cells"]
        if cell["cell_type"] == "code" and "33_rerank_retrieval.py" in "".join(cell["source"])
    )
    assert "30_retrieve_questions.py" not in rerank_cell
    assert '"--candidate-tables"' in code and '"100"' in code
    assert '"--max-length"' in code and '"8192"' in code
    assert '"--max-batch-tokens"' in code and '"8192"' in code
    assert "retrieval_rerank_shards/shard_0/run_metadata.json" in code
    assert "checkpoint_project_revision" in code
    assert '"hybrid_project_revision": HYBRID_PROJECT_SHA' in code
    assert '"reranker_project_revision": RERANK_PROJECT_SHA' in code
    # A final run may legitimately use the hybrid ranking, but never an unrecorded one.
    assert "if RETRIEVAL != HYBRID:" in code
    assert "The retrieval used by a final run must be recorded." in code
    assert '"--data-parallel-size"' in code
    assert '"--shard-count"' in code
    # hybrid retrieval, reranking, D1, smoke, widest-route, sample, unit ablation, full run
    assert code.count('"--project-revision"') == 8
    assert '"scripts/51_merge_generation_shards.py"' in code
    assert 'iter_input_paths(f"{GENERATION_NAME}_shards/shard_0/run_metadata.json")' in code
    assert "expected_smoke_answers = {1: 208253.201298, 213: 6.15569834}" in code
    assert "Smoke used a fallback answer" in code
    assert 'SMOKE_GATE = SMOKE_GEN / "smoke_gate.json"' in code
    assert '"projected_unit_branch_hours": projected_unit_branch_hours' in code
    assert 'os.environ["VIFINQA_ALLOW_LONG_SUBSET"] = "0"' in code
    assert "Smoke projects {projected_branch_hours:.2f}h for one branch" in code
    assert "VLLM_CONFIG" in code
    assert "cuda_driver_linker_environment" in code
    assert 'linker_name = linker_dir / "libcuda.so"' in code
    assert 'for variable in ("LIBRARY_PATH", "LD_LIBRARY_PATH")' in code
    assert "env=server_environment" in code
    assert "[-50_000:]" in code
    assert '"scripts/45_finalize_submission.py"' in code
    assert 'FINAL_SUBMISSION = GEN / "submission_z4_abs.json"' in code
    assert '"submission_profile": "z4_abs_line"' in code
    assert code.count('"--allow-partial-docs"') == 2
    assert "Qwen2.5" not in code


def test_notebooks_discover_inputs_through_symlinked_kaggle_mounts() -> None:
    module_source = KAGGLE_INPUTS.read_text(encoding="utf-8")
    helpers = module_source[module_source.index("def iter_input_paths") :].rstrip("\n")
    for notebook in (BUILDER_NOTEBOOK, GENERATION_NOTEBOOK, RESUME_NOTEBOOK, PACKAGE_NOTEBOOK):
        code = _compiled_code(notebook)
        # The bootstrap cell runs before the project is installed, so it carries a verbatim
        # copy of the module instead of importing it.
        assert helpers in code, f"{notebook.name} must copy src/vifinqa/kaggle_inputs.py verbatim"
        # rglob is fine on the working directory; what it cannot do is expand `**` across
        # the symlinks Kaggle mounts inputs behind.
        for forbidden in ('Path("/kaggle/input").rglob', "INPUT_ROOT.rglob"):
            assert forbidden not in code, f"{notebook.name} must not {forbidden}"
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


def test_resume_notebook_finishes_a_run_without_rebuilding_its_retrieval() -> None:
    code = _compiled_code(RESUME_NOTEBOOK)

    # Resuming needs neither the dense index nor the stages that produced the retrieval.
    assert "22_build_dense.py" not in code
    assert "30_retrieve_questions.py" not in code
    assert "33_rerank_retrieval.py" not in code

    # The generation checkpoint is keyed to the retrieval's SHA-256 and to the commit that
    # wrote it, so both are taken from the checkpoint rather than recomputed or assumed.
    assert 'iter_input_paths("retrieval_reranked.jsonl")' in code
    assert 'iter_input_paths("retrieval_hybrid.jsonl")' in code
    assert 'sha256(path) == prior_metadata.get("retrieval_sha256")' in code
    assert 'PROJECT_SHA = prior_metadata["project_revision"]' in code
    assert 'TABLE_UNIT_SOURCE = str(prior_metadata.get("table_unit_source", "latest"))' in code
    assert '"--table-unit-source",\n    TABLE_UNIT_SOURCE,' in code
    assert '"checkout", PROJECT_SHA' in code
    assert "assert checked_out == PROJECT_SHA" in code

    # Kaggle mounts inputs read-only and copytree preserves that.
    assert "def make_writable(" in code
    assert "make_writable(GEN_SHARDS)" in code

    # A shard count that disagrees with the checkpoint would silently start over.
    assert "shard_count == SHARDS" in code
    assert "resuming with {completed_rows()}/1012 questions already answered" in code
    assert "if answered < 1012:" in code

    assert "Qwen/Qwen3-14B-AWQ" in code
    assert "31c69efc29464b6bb0aee1398b5a7b50a99340c3" in code
    assert "Qwen/Qwen3-8B-AWQ" in code
    assert "4da05a8edb55c6046cce958586c33b61da07bb79" in code
    assert "Attach exactly one supported generation checkpoint" in code
    assert '"--final-run"' in code
    # A session that stops at a cap must not package a submission that is not finished.
    assert 'os.environ["VIFINQA_QUESTION_LIMIT"]' in code
    assert 'common_cmd += ["--limit", QUESTION_LIMIT]' in code
    assert "skipping merge and packaging" in code
    # Fetching a bare SHA is a server permission, not a guarantee; a full clone has it.
    assert '"fetch", "origin", PROJECT_SHA], check=False' in code
    assert '"scripts/45_finalize_submission.py"' in code
    assert 'FINAL_SUBMISSION = GEN / "submission_z4_abs.json"' in code
    assert code.count('"--allow-partial-docs"') == 2
    assert '"scripts/41_package_submission.py"' in code


def test_packaging_notebook_needs_no_accelerator_and_downloads_one_file() -> None:
    code = _compiled_code(PACKAGE_NOTEBOOK)

    # Merging, re-executing and zipping are CPU work. Asking for a GPU here would spend
    # quota to do arithmetic, and pulling the serving stack would spend minutes on it.
    assert "torch" not in code
    assert "vllm" not in code
    assert "requirements-gpu.txt" not in code
    assert 'str(PROJECT / "requirements.txt")' in code
    assert "Qwen/Qwen3-14B-AWQ" in code
    assert "Qwen/Qwen3-8B-AWQ" in code
    assert "Attach exactly one supported finished checkpoint" in code
    assert 'GENERATION_NAME = f"generation_{MODEL_PROFILE}"' in code
    assert 'file_sha256(path) == prior_metadata.get("retrieval_sha256")' in code

    # It packages with the commit that produced the answers, not with whatever main is.
    assert '"checkout", PROJECT_SHA' in code
    assert '"fetch", "origin", PROJECT_SHA], check=False' in code

    assert '"scripts/52_restore_evidence_csv.py"' in code
    assert '"scripts/51_merge_generation_shards.py"' in code
    assert '"scripts/45_finalize_submission.py"' in code
    assert 'FINAL_SUBMISSION = GEN / "submission_z4_abs.json"' in code
    assert '"scripts/40_validate_submission.py"' in code
    assert '"scripts/41_package_submission.py"' in code
    assert code.count('"--allow-partial-docs"') == 2
    assert "assert answers == 1012" in code
    assert "Download only this ZIP." in code
