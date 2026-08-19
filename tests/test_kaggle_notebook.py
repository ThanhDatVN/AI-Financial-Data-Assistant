from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER_NOTEBOOK = ROOT / "notebooks/01_kaggle_build_dense_artifact.ipynb"
GENERATION_NOTEBOOK = ROOT / "notebooks/02_kaggle_dense_and_generate.ipynb"
RESUME_NOTEBOOK = ROOT / "notebooks/03_kaggle_resume_and_submit.ipynb"
PACKAGE_NOTEBOOK = ROOT / "notebooks/04_kaggle_package_submission.ipynb"
SYNTHETIC_NOTEBOOK = ROOT / "notebooks/05_kaggle_render_and_filter_synthetic.ipynb"
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
    # Three rankings, one scored run, one submission each: BM25 alone put a gold table in the
    # prompt for 0.6168 of questions, BM25 plus dense fusion for 0.5997, and both plus the
    # cross-encoder for 0.5642 (3135/3136/3137). Every stage on top of BM25 made it worse and
    # cost GPU hours, so the default has to be the one that measured best, and the two losing
    # stages have to be skippable rather than required.
    assert 'RANKING = os.environ.get("VIFINQA_RANKING", "bm25")' in code
    assert 'os.environ["VIFINQA_RANKING"] = "bm25"' in code
    assert "RETRIEVAL = SPARSE" in code
    assert 'if RANKING == "hybrid":' in code
    # `RERANK` carries the whole answer, because the run manifest reads it at the end of a
    # twelve-hour run to decide whether a reranker revision exists to record. Asking for the
    # reranker on a ranking it cannot reorder left the name true and its revision undefined.
    assert 'RERANK = os.environ.get("VIFINQA_RERANK") == "1" and RANKING == "hybrid"' in code
    assert "if not RERANK:" in code
    # Assume the consolidated statement when the question is silent: the candidates it frees are
    # a quarter of the prompt, and the same flag has to reach every generation call site or a
    # resume is refused for a setting nobody chose.
    assert code.count('"--scope-router",') == 5
    # Measured on 18/08: citing the candidate set under this policy scored table recall 0.6922
    # against 0.6168 without it, and MRR@5 0.4195 against 0.3476 (3165 vs 3137). It is on by
    # default because it costs no context -- it drops the other statement's tables and refills
    # the slots from further down the same ranking.
    assert 'os.environ["VIFINQA_SCOPE_ROUTER"] = "consolidated"' in code
    # And the run finishes in one session. A non-empty limit -- "1012" included -- takes the
    # partial-run branch, which stops without packaging.
    assert 'os.environ["VIFINQA_QUESTION_LIMIT"] = ""' in code
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
    # Kaggle mounts inputs read-only and copytree preserves that, so an imported checkpoint is
    # imported successfully and then fails on its first write. Both call sites need the fix.
    assert "def make_writable(" in code
    assert code.count("make_writable(") == 3
    assert '"bm25_project_revision": SPARSE_PROJECT_SHA' in code
    assert '"hybrid_project_revision": (HYBRID_PROJECT_SHA if HYBRID is not None else None)' in code
    assert '"reranker_project_revision": (RERANK_PROJECT_SHA if RERANK else None)' in code
    # A final run may legitimately use any of the three rankings, but never an unrecorded one.
    assert "if RETRIEVAL not in (SPARSE, HYBRID):" in code
    assert "The retrieval used by a final run must be recorded." in code
    assert '"--data-parallel-size"' in code
    assert '"--shard-count"' in code
    # BM25 retrieval, hybrid retrieval, reranking, D1, smoke, widest-route, sample, unit
    # ablation, full run
    assert code.count('"--project-revision"') == 9
    assert '"scripts/51_merge_generation_shards.py"' in code
    assert 'iter_input_paths(f"{GENERATION_NAME}_shards/shard_0/run_metadata.json")' in code
    assert "expected_smoke_answers = {1: 208253.201298, 213: 6.15569834}" in code
    # The smoke gate names the questions it tolerates rather than counting them. Five IDs
    # chosen as the hardest in the release make a count say more about the sample than the run;
    # a fourth ID appearing is what a regression actually looks like.
    assert "KNOWN_HARD = {399, 442, 473}" in code
    assert "unexpected = sorted(" in code
    assert "assert not unexpected" in code
    assert 'trace.get("rescued")' in code
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


def test_synthetic_notebook_measures_x_and_refuses_a_set_it_cannot_phrase() -> None:
    """The first measurement of the third factor, and the two ways it can quietly be wrong.

    A Hard sample carries a second line item that decides which years count. Rendered without it,
    the question asks the plainer thing, whose answer the sampler had already proved differs -- so
    the whole 21.3% tier would read as "the model finds Hard hard" when it was never asked.

    And gold-only prompts measure the solver with the retrieval difficulty removed. That is the
    right number for comparing two prompts and the wrong one for deciding what a GPU is worth, so
    the notebook has to say which one it just printed.
    """
    code = _compiled_code(SYNTHETIC_NOTEBOOK)

    assert '"scripts/71_render_questions.py"' in code
    assert '"scripts/73_filter_synthetic.py"' in code
    # Both stages run against the served model, and neither needs the dense index or a submission.
    assert "22_build_dense.py" not in code
    assert "33_rerank_retrieval.py" not in code
    assert "41_package_submission.py" not in code

    assert 'unphrasable = [row for row in hard if not row.get("condition")]' in code
    assert "assert not unphrasable" in code
    # And it has to point at the backfill, not the sampler: `--split dev --count 499` now returns
    # 416 samples with 23 Hard ones against the 106 on disk, so re-sampling to add a field would
    # swap the set that mirrors the paper for one that does not.
    assert "scripts/74_backfill_condition.py" in code
    assert "Do NOT " in code and "re-run 70_sample_programs.py" in code

    # Attempt 1 is greedy and mirrors production; later attempts sample at 0.7. Dividing total
    # hits by total attempts averaged two decoding regimes and called the result the
    # production-comparable one. The greedy figure is the one to carry to the scoreboard.
    assert 'int(row["attempt_hits"][0])' in code
    assert "greedy = sum(first_try) / len(first_try) if first_try else None" in code
    assert '"x_greedy_attempt_1": greedy' in code
    # And the per-family denominator, without which the total cannot be re-weighted afterwards.
    assert '"attempts": sum(' in code
    assert "pass_at_1 = hits / tries" not in code
    assert "upper bounds on production X" in code
    assert '"--distractors",' in code

    # `outputs/` is outside git, so the programs arrive as a Dataset and the results have to be
    # downloaded. A session that loses either has nothing to show for its hours.
    assert "Upload `outputs/synthetic/` as a Kaggle Dataset" in code
    assert "outputs/ is outside git, so this is the only copy." in code
    # Twelve-hour cap: both stages append and skip what they judged, so a second session finishes.
    assert "resuming from" in code

    # The filter costs 8-9 hours, so a mangled render has to stop the session before it starts
    # rather than after. The floor is overridable, because a deliberate --limit run trips it.
    assert 'RENDER_FLOOR = float(os.environ.get("VIFINQA_RENDER_FLOOR", "0.85"))' in code
    assert "assert rendered_share >= RENDER_FLOOR" in code
    assert "VIFINQA_RENDER_FLOOR=0" in code


def test_notebooks_discover_inputs_through_symlinked_kaggle_mounts() -> None:
    module_source = KAGGLE_INPUTS.read_text(encoding="utf-8")
    helpers = module_source[module_source.index("def iter_input_paths") :].rstrip("\n")
    for notebook in (
        BUILDER_NOTEBOOK,
        GENERATION_NOTEBOOK,
        RESUME_NOTEBOOK,
        PACKAGE_NOTEBOOK,
        SYNTHETIC_NOTEBOOK,
    ):
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
    # A BM25-only run leaves a differently named file, and a resume that cannot find the
    # retrieval its checkpoint was built against cannot resume anything.
    assert 'iter_input_paths("retrieval_bm25.jsonl")' in code
    assert 'sha256(path) == prior_metadata.get("retrieval_sha256")' in code
    assert 'SCOPE_ROUTER = str(prior_metadata.get("scope_router", "both"))' in code
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


def test_both_generation_notebooks_pass_every_flag_the_fingerprint_reads() -> None:
    """The second notebook finishes what the first started, and a fingerprint mismatch refuses it.

    Every generator flag in the fingerprint has to be passed by both, or the resume is rejected
    for a setting nobody chose to change. `--max-tokens` was the one that got away: the first
    notebook passed 4096 and the second left it to the script's default, so the checkpoint could
    never be picked up. Comparing the two call sites catches the next one without naming it.
    """
    fingerprinted = {
        "--model",
        "--model-revision",
        "--thinking-mode",
        "--table-unit-source",
        "--candidate-tables",
        "--scope-router",
        "--max-tokens",
        "--context-limit",
        "--max-attempts",
        "--project-revision",
    }
    generation = _compiled_code(GENERATION_NOTEBOOK)
    resume = _compiled_code(RESUME_NOTEBOOK)
    missing = sorted(flag for flag in fingerprinted if f'"{flag}",' not in resume)
    assert not missing, f"resume notebook never passes {missing}"
    unpassed = sorted(flag for flag in fingerprinted if f'"{flag}",' not in generation)
    assert not unpassed, f"generation notebook never passes {unpassed}"

    # And the two budget figures must agree with the server they are sized against, which is why
    # one constant feeds both rather than two literals that can drift apart.
    assert "MAX_MODEL_LEN = 16384" in generation
    assert '"max_model_len": MAX_MODEL_LEN,' in generation
    assert '"--context-limit",\n    str(MAX_MODEL_LEN),' in generation
    assert 'CONTEXT_LIMIT = str(prior_metadata["context_limit"])' in resume
    assert 'MAX_TOKENS = str(prior_metadata["max_tokens"])' in resume


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


def test_a_default_full_run_is_not_stopped_by_a_cell_it_does_not_use() -> None:
    """RUN_MODE="full" must reach the run without a hand edit.

    Twice now a 12-hour session halted between the smoke and the run it was preparing for: once
    on the opt-in 200-question ablation, which a full run has no business entering, and once on a
    hand-pasted commit SHA that cells 2 and 3 already establish by fetching the ref. Both were
    guards on things the run does not use.

    So evaluate every assert whose condition is made only of the run flags and literals, under
    exactly what cell 1 sets for a full run and what a fresh session leaves alone. Anything that
    fails here would stop the session there.
    """
    flags = {
        "RUN_FULL": True,
        "FINAL_RUN": True,
        "RUN_SUBSET_200": False,
        "RUN_DIAGNOSTICS": False,
        # The pinned profile, so the 14B eligibility branch is evaluated rather than assumed.
        "MODEL_TOTAL_PARAMS_B": 8.2,
        "THINKING_MODE": "disabled",
    }
    environment = {
        "VIFINQA_RUN_FULL": "1",
        "VIFINQA_FINAL_RUN": "1",
        "VIFINQA_RUN_SUBSET_200": "0",
        "VIFINQA_RUN_DIAGNOSTICS": "0",
        "VIFINQA_ALLOW_LONG_SUBSET": "0",
        "VIFINQA_EXPECTED_PROJECT_SHA": "",
        "VIFINQA_QUESTION_LIMIT": "600",
        "VIFINQA_ORGANIZER_CONFIRMED_14B": "1",
        "VIFINQA_THINKING_MODE": "disabled",
    }

    class _Environ(dict[str, str]):
        def get(self, key: str, default: str = "") -> str:  # type: ignore[override]
            return environment.get(key, default)

    namespace: dict[str, object] = {**flags, "os": type("os", (), {"environ": _Environ()})()}

    notebook = json.loads(GENERATION_NOTEBOOK.read_text(encoding="utf-8"))
    stopped: list[tuple[int, str]] = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        tree = ast.parse(source)
        skipped = _skipped_statements(tree, namespace)
        for statement in ast.walk(tree):
            if not isinstance(statement, ast.Assert) or statement in skipped:
                continue
            try:
                holds = bool(
                    eval(compile(ast.Expression(statement.test), "<test>", "eval"), namespace)
                )  # noqa: S307
            except NameError:
                continue  # depends on the machine, not on the configuration
            if not holds:
                stopped.append((index, (ast.get_source_segment(source, statement) or "")[:90]))
    assert not stopped, f"a default full run would stop here: {stopped}"


def _skipped_statements(tree: ast.AST, namespace: dict[str, object]) -> set[ast.stmt]:
    """Statements inside a branch the default configuration never enters."""
    skipped: set[ast.stmt] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        try:
            taken = bool(eval(compile(ast.Expression(node.test), "<test>", "eval"), namespace))  # noqa: S307
        except (NameError, AttributeError, TypeError):
            continue
        for statement in node.orelse if taken else node.body:
            skipped.update(inner for inner in ast.walk(statement) if isinstance(inner, ast.stmt))
    return skipped


def test_two_attached_sessions_can_be_told_apart_instead_of_detached() -> None:
    """A finished run and the one before it are both legitimately attached at once.

    The earlier session holds the hybrid ranking and the diagnostics the later one no longer
    writes, so refusing outright forces the user to detach an input they still need. Naming the
    one you mean is the way through; the guard still refuses when nothing is named and more than
    one is present, because resuming the wrong run silently is worse than stopping.
    """
    for notebook in (RESUME_NOTEBOOK, PACKAGE_NOTEBOOK):
        code = _compiled_code(notebook)
        assert 'os.environ.get("VIFINQA_CHECKPOINT_SOURCE", "").strip()' in code
        assert "if WANTED_CHECKPOINT:" in code
        assert "WANTED_CHECKPOINT in str(candidate[1])" in code
        # The guard survives: naming nothing with two attached still stops the run.
        assert "assert len(checkpoint_candidates) == 1" in code
        # And the message says how to get past it, which is the half that was missing.
        assert "or name one with VIFINQA_CHECKPOINT_SOURCE" in code


def _branch_only_bindings(path: Path) -> dict[str, int]:
    """Names a later cell reads that an earlier cell only ever bound inside a branch.

    `test_a_default_full_run_is_not_stopped_by_a_cell_it_does_not_use` walks the same notebook
    but swallows `NameError`, on the reasoning that an unbound name there depends on the machine
    rather than on the configuration. That exemption is exactly how `KERNEL_DIAGNOSTIC` reached
    the final manifest: cell 8 bound it only under `VIFINQA_RUN_DIAGNOSTICS=1`, and the manifest
    read it in the condition of its own `.exists()` guard, so no guard could save it. It fires
    only when this notebook packages its own submission, which nothing did until the run was
    uncapped -- so it would have raised after nine hours, with `submission.zip` already written.
    """
    notebook = json.loads(path.read_text(encoding="utf-8"))
    top_level: set[str] = set()
    branch_only: dict[str, int] = {}
    found: dict[str, int] = {}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in ast.walk(tree):
            read = isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            if read and node.id in branch_only and node.id not in top_level:
                found.setdefault(node.id, branch_only[node.id])
        for statement in tree.body:
            unconditional = isinstance(
                statement, ast.Assign | ast.AnnAssign | ast.AugAssign | ast.For | ast.With
            )
            for node in ast.walk(statement):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    if unconditional:
                        top_level.add(node.id)
                    else:
                        branch_only.setdefault(node.id, index)
            if isinstance(statement, ast.FunctionDef | ast.ClassDef):
                top_level.add(statement.name)
    return found


def test_the_final_manifest_reads_no_name_a_skipped_diagnostic_would_have_left_unbound() -> None:
    tracked = {"PERF_DIAGNOSTIC", "KERNEL_DIAGNOSTIC", "SMOKE_GATE"}
    for notebook in (GENERATION_NOTEBOOK, SYNTHETIC_NOTEBOOK):
        leaked = _branch_only_bindings(notebook)
        assert not (tracked & leaked.keys()), (
            f"{notebook.name} reads {sorted(tracked & leaked.keys())} in a later cell, "
            "but an earlier cell binds it only inside a branch the default run skips"
        )


def test_the_render_call_leaves_the_token_budget_to_the_script() -> None:
    """Both ceilings were raised together; only one of them reached this call site.

    `71_render_questions.py` defaults to 640 tokens because its schema now allows a 700-character
    question, and a Hard question carries a filter clause and an answer clause in one sentence.
    The notebook went on passing 256 -- roughly 400 Vietnamese characters, which is exactly where
    round two's refused generations piled up against the old ceiling. A budget that runs out
    mid-string returns unterminated JSON, and that path counts the refusal without keeping its
    text, so the next diagnosis would have been back to reading counts.
    """
    notebook = json.loads(SYNTHETIC_NOTEBOOK.read_text(encoding="utf-8"))
    render_cells = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "71_render_questions.py" in "".join(cell["source"])
    ]
    assert len(render_cells) == 1
    code = "\n".join(
        line for line in render_cells[0].splitlines() if not line.strip().startswith("#")
    )
    assert '"--max-tokens"' not in code, "let 71's own 640-token default stand"
    # The retry carries the reason now, so a third attempt buys a corrected draw rather than a
    # warmer redraw of the same mistake.
    assert '"--attempts",\n    "3",' in code
    assert '"--rejected",' in code


def test_the_greedy_figure_skips_rows_written_before_the_field_existed() -> None:
    """A session resumed across this change holds both kinds of row in one file.

    `73` gained `attempt_hits` while a split was already being filtered, so the output a second
    session appends to can hold rows that predate it. Averaging the two together would quietly
    mix a greedy-only figure with rows that have no greedy figure at all, so the reader has to
    drop the older rows and say how many it dropped. The comprehension is lifted out of the
    notebook rather than restated here, so the two cannot drift apart.
    """
    notebook = json.loads(SYNTHETIC_NOTEBOOK.read_text(encoding="utf-8"))
    cell = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "first_try = [" in "".join(cell["source"])
    )
    lines = cell.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("first_try = ["))
    end = next(i for i, line in enumerate(lines[start:], start) if line == "]")
    block = chr(10).join(lines[start : end + 1])

    judged = [
        {"attempt_hits": [1, 0]},  # right on the greedy try
        {"attempt_hits": [0, 1]},  # only the sampled retry got it
        {"attempt_hits": [0, 0]},
        {"solved_attempts": 1, "of_attempts": 2},  # written before the field existed
        {"attempt_hits": []},  # setup failed before any attempt ran
    ]
    namespace: dict[str, object] = {"judged": judged}
    exec(block, namespace)  # noqa: S102 - the notebook's own line, run on fixture rows
    first_try = namespace["first_try"]
    assert first_try == [1, 0, 0], "index 0 is the greedy attempt, and only rows that have one"
    assert len(first_try) < len(judged), "the legacy row has to be excluded, not defaulted to 0"


def test_the_filter_is_sharded_and_carries_both_measured_switches() -> None:
    """One client against an eight-wide server is what made a 499-sample measurement cost 9.4h.

    Notebook 02 has driven the generator with `--shard-count`/`--shard-index` since the first full
    run; the dev bench never got the same treatment, and every experiment about X paid for it. The
    two switches are here for the same reason `--scope-router` is in notebook 02: measured, off by
    default, and named in the summary so a result can say which setting produced it.
    """
    code = _compiled_code(SYNTHETIC_NOTEBOOK)

    assert "FILTER_SHARDS = 8" in code
    assert '"--shard-count",' in code and '"--shard-index",' in code
    assert "subprocess.Popen(shard_cmd)" in code
    # A shard that dies has to stop the notebook, not leave a hole in the denominator.
    assert "assert not failed" in code
    # Per-shard files: one handle shared across processes would interleave half-written lines.
    assert "SOLVED_SHARDS = [" in code and "REJECTED_SHARDS = [" in code
    assert "read_rows(path) for" not in code  # the union is built explicitly, not by glob luck
    assert "for path in SOLVED_SHARDS" in code

    assert 'ROOT_GRAMMAR = "off"' in code
    assert "WORKED_EXAMPLE = False" in code
    assert '"--root-grammar",' in code
    assert 'filter_cmd += ["--worked-example"]' in code
    # Recorded with the result, or the number cannot say what produced it.
    assert '"root_grammar": ROOT_GRAMMAR' in code
    assert '"worked_example": WORKED_EXAMPLE' in code
    # The miss taxonomy used to exist only in one process's stdout.
    assert '"misses": miss_totals' in code


def test_a_second_setting_writes_its_own_files_instead_of_finding_them_done() -> None:
    """`73` skips any id its output already holds -- right for resuming, wrong for comparing.

    The dev bench exists to run one setting against another. With the verdict files named after
    the split alone, running "off" and then "year" would leave the second pass with every id
    already judged: it would write nothing, print the first pass's numbers, and look like a
    result. The questions stay named after the split, because they are the half worth reusing.
    """
    code = _compiled_code(SYNTHETIC_NOTEBOOK)
    assert 'RUN_TAG = f"{SPLIT}_{ROOT_GRAMMAR}" + ("_ex" if WORKED_EXAMPLE else "")' in code
    for artefact in ("_solved.s{index}.jsonl", "_rejected.s{index}.jsonl", "_x_measurement.json"):
        assert f"{{RUN_TAG}}{artefact}" in code, artefact
    # The rendered questions are the reusable asset and must NOT be keyed to the solver setting.
    assert 'f"{SPLIT}_questions.jsonl"' in code
    assert 'f"{RUN_TAG}_questions.jsonl"' not in code
