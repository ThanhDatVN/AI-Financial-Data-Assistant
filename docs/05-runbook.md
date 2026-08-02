# Runbook tái lập

## 1. Môi trường local

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -e . --no-deps --no-build-isolation
```

Kiểm dependency:

```powershell
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe -c "import pandas, pyarrow, torch; print(pandas.__version__, pyarrow.__version__, torch.__version__)"
```

## 2. Tải data

```powershell
.venv\Scripts\huggingface-cli.exe download AIGuruTinix/ViFinQA `
  --repo-type dataset --local-dir data\raw\ViFinQA
```

Xác nhận có `questions/questions.jsonl`, `code_stock.csv`, `financial_statements/`.

## 3. Audit, manifest, index

```powershell
.venv\Scripts\python.exe scripts\00_audit_dataset.py
.venv\Scripts\python.exe scripts\10_build_manifest.py --workers 6
.venv\Scripts\python.exe scripts\20_build_bm25.py
.venv\Scripts\python.exe scripts\30_retrieve_questions.py --candidate-k 2000
.venv\Scripts\python.exe scripts\32_validate_retrieval.py
```

Expected release hiện tại:

- audit 1.012 questions, 1.973 docs, 146.246 tables;
- manifest metadata `tables=146246`;
- unique table refs 146.246;
- retrieval rows 1.012; QC `passed=true`, không empty/ref sai/route khả dụng bị thiếu.

## 4. Smoke evidence

```powershell
.venv\Scripts\python.exe scripts\31_export_evidence.py `
  "VJC_financial_statements_2018_separate|table_50" `
  --output data\interim\evidence
```

Kiểm row “Lãi tiền gửi”, column “2018”, raw `208.253.201.298`; integration test phải tính
`208253.201298` triệu đồng.

## 5. Test gates

```powershell
.venv\Scripts\ruff.exe check src scripts tests
.venv\Scripts\ruff.exe format --check src scripts tests
.venv\Scripts\mypy.exe src scripts
.venv\Scripts\python.exe -m pytest -q
```

Test integration tự skip khi không có corpus/manifest.

Tạo template qrels nội bộ (không phải gold):

```powershell
.venv\Scripts\python.exe scripts\60_sample_qrels.py --size 100
.venv\Scripts\python.exe scripts\61_build_qrels_pool.py `
  --run bm25=outputs\retrieval.jsonl `
  --depth 20
```

Pool BM25 trên chỉ là smoke. Sau GPU run, chạy lại với `--run hybrid=...` và các run late/rerank đã
khóa. Chỉ tính metric sau khi pool đủ đa dạng, hai annotator độc lập, adjudication hoàn tất và các dòng
qrels có `status=adjudicated`. Validate bằng `configs/qrels.schema.json` và
`configs/qrels_pool.schema.json`; archive cả hai hash.

## 6. GPU

```bash
python -m pip install -r requirements-gpu.txt
python scripts/22_build_dense.py --device cuda --batch-size 16 \
  --model-revision <bge-m3-pre-cutoff-sha> --final-run
python scripts/30_retrieve_questions.py --dense data/index/bge_m3
vllm serve Qwen/Qwen2.5-Coder-7B-Instruct-AWQ \
  --max-model-len 8192 --seed 20260802
python scripts/50_generate_programs.py --limit 5 \
  --model-revision <qwen-awq-pre-cutoff-sha> \
  --execution-timeout 10 --memory-limit-mb 4096
```

Chỉ chạy full sau khi smoke 5 câu không có server/schema/execution error.
`--memory-limit-mb` cần POSIX; bỏ flag này trên Windows. Dù có process timeout, runtime cuối vẫn cần
container/read-only filesystem/no-network policy nếu xem model output là đối thủ.

## 7. Validation/package

```powershell
.venv\Scripts\python.exe scripts\40_validate_submission.py `
  outputs\generation\submission.json `
  --evidence-root outputs\generation

.venv\Scripts\python.exe scripts\41_package_submission.py `
  outputs\generation\submission.json submissions\submission.zip `
  --evidence-root outputs\generation
```

Package command từ chối overwrite nếu thiếu `--force`. Không dùng `--force` với ZIP đã nộp; tạo tên
run mới và giữ SHA-256.

## 8. Cấu trúc artefact run

```text
runs/RUN_ID/
├── config.yaml
├── environment.json
├── data_manifest.json
├── retrieval.jsonl
├── predictions.jsonl
├── errors.jsonl
├── summary.json
├── submission.json
└── submission.sha256
```

`RUN_ID` nên gồm UTC timestamp + git short SHA + config name. Không ghi token/secret vào config.

## 9. Khôi phục lỗi

- Manifest dừng: chạy lại; output chưa freeze được ghi mới. Muốn resume cần triển khai shard/checkpoint.
- Dense dừng: notebook hiện build lại; ưu tiên bổ sung embedding shard trước full GPU run.
- Generation dừng: chạy lại cùng output, script resume theo prediction ID.
- Validation mismatch: không sửa `answer` thủ công; sửa query/evidence rồi execute lại.
- Source hash drift: ngừng, xác nhận data revision; không bỏ hash check.
- Table ref dashboard fail: đổi config/formatter, không đổi internal ordinal/provenance.
