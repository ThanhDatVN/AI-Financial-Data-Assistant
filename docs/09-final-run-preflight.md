# Final-run preflight và Kaggle handoff

**Local preflight:** `RUN-012`, 02/08/2026. Không có metric accuracy vì qrels còn unlabeled.

## 1. Kết quả local đã chạy

| Gate | Kết quả |
|---|---|
| BM25 full rerun | 1.012 câu, 55,561 giây, byte-identical frozen v2 |
| retrieval SHA-256 | `99bcd3646952d20afc5fa201c9308a05b75c800fac7399287ec34d9d0538bf09` |
| retrieval QC | `passed=true`; mọi failure count bằng 0 |
| dense CPU compatibility | BGE-M3 offline, 100 bảng, 147,3 giây |
| dense positive route smoke | q471 trả 5 hit `AAA–2016–consolidated` |
| qrels pool control | 100 câu/1.998 candidate; chưa phải diverse pool |

`outputs/retrieval_balanced.jsonl` là artefact cũ. Frozen retrieval v2 dùng cho handoff là
`outputs/retrieval.jsonl`; không dùng file có hậu tố `balanced` để tạo pool hoặc final run.

## 2. Revisions đã khóa

| Stage | Model | Full revision | Ngày commit | License/status |
|---|---|---|---|---|
| dense | `BAAI/bge-m3` | `5617a9f61b028005a4858fdac845db406aefb181` | 03/07/2024 | MIT, public |
| rerank candidate | `BAAI/bge-reranker-v2-m3` | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` | 24/06/2024 | Apache-2.0, public |
| generator | `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ` | `8e8ed243bbe6f9a5aff549a0924562fc719b2b8a` | 18/11/2024 | Apache-2.0, public |

Nguồn kiểm: commit history chính chủ của
[BGE-M3](https://huggingface.co/BAAI/bge-m3/commits/main),
[BGE reranker](https://huggingface.co/BAAI/bge-reranker-v2-m3/commits/main) và
[Qwen AWQ](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-AWQ/commits/main). Full SHA được
đối chiếu qua Hugging Face model revision API. File `configs/final_run.env.example` chứa đúng các SHA.

## 3. Artefact mang lên Kaggle

1. current Git checkout, không dùng archive thiếu `.git` cho final;
2. `data/raw/ViFinQA`;
3. `data/processed/table_manifest.jsonl` và `.parquet`;
4. `data/index/bm25/`;
5. `outputs/retrieval.jsonl` + `outputs/retrieval_qc.json` làm sparse control;
6. `annotations/qrels_template.jsonl` + control pool để tạo diverse pool sau dense;
7. `requirements-gpu.txt` và notebook 02.

Đối chiếu SHA bằng `runs/20260802_RUN012_local_preflight/handoff_manifest.json`. Kaggle phải fail nếu
manifest/data/retrieval hash lệch. Handoff manifest hiện có SHA-256
`fef7b2cb25a2d8c8bd718110cffd954b8a1ed2c40ea0b3c4e8136092b3e63a4b`.

## 4. Thứ tự chạy trên Kaggle

1. Bật GPU và attach repo/data/artefacts.
2. Mở notebook 02; config cell đã pin model/dense revision, `FINAL_RUN=0`, `RUN_FULL=0`, `TP=1`.
   Làm theo checklist cell-by-cell tại `docs/10-huong-dan-chay-thu-nghiem.md`.
3. Chạy hardware/install/path cells; lưu `runtime_environment.txt`.
4. Build full BGE-M3 index; kiểm config ghi đúng revision và 146.246 tables.
5. Chạy hybrid retrieval + QC; tải/checkpoint artefacts trước khi khởi động vLLM.
6. Build diverse pool từ sparse + hybrid; chưa gán metric nếu chưa adjudicate.
7. Khởi động Qwen/vLLM; chạy 5-question generation smoke.
8. Kiểm `errors.jsonl`, IR, selected cells, units, traces và re-execution.
9. Chỉ sau khi smoke pass: đặt `VIFINQA_RUN_FULL=1`; nếu là submission candidate đã xác nhận rule,
   đặt thêm `VIFINQA_FINAL_RUN=1`, rerun config cell rồi chạy thẳng full cell. Không rerun vLLM cell.
10. Validate 1.012/1.012, package, lưu ZIP hash; không upload khi table-ref/tolerance contract chưa xác nhận.

## 5. Final stop conditions

- model/dense revision khác SHA đã khóa hoặc dùng `main`;
- project không có Git SHA;
- dense config count khác 146.246 hoặc manifest hash lệch;
- retrieval QC không `passed=true`;
- smoke có server/schema/grounding/unit/execution error chưa phân loại;
- full generation thiếu ID hoặc có silent fallback;
- validator/package fail;
- dashboard table-ref/tolerance vẫn unresolved cho submission thật.
