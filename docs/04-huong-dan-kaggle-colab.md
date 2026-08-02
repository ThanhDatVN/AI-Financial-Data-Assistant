# Hướng dẫn Colab/Kaggle

## 1. Chọn runtime nào

- Local/Colab CPU: audit, manifest, BM25, validation, packaging.
- Kaggle/Colab GPU: BGE-M3, reranker, Qwen/vLLM.
- vLLM chạy Linux; tài liệu chính thức không hỗ trợ Windows native. Local Windows chỉ gọi endpoint
  OpenAI-compatible ở WSL/Linux/host khác.

Google Colab công bố tài nguyên, GPU type, timeout và quota thay đổi; free runtime thường tối đa 12
giờ tùy khả dụng/usage. Kaggle cũng có quota/runtime policy biến động. Không lập kế hoạch dựa trên
một loại GPU hoặc số giờ cứng.

## 2. Artefact cần mang giữa runtime

| Artefact | Kích thước thực tế hiện tại | Tái sinh được |
|---|---:|---|
| raw corpus | 362,6 MiB | tải HF |
| manifest JSONL | 426,4 MB | có |
| manifest Parquet | 50,1 MB | có |
| BM25 index + metadata | 170,8 MB | có |
| dense FAISS | phụ thuộc dimension | có, tốn GPU |
| predictions/evidence | phụ thuộc top evidence | checkpoint |
| qrels v2 + pool control | ~0,60 MB | có; chưa có human gold |

Ưu tiên upload Parquet + BM25 + raw corpus. JSONL cần cho script dense hiện tại; nếu storage hạn chế,
có thể bổ sung reader Parquet rồi bỏ JSONL.

## 3. Colab preparation

Mở [`01_colab_prepare_and_index.ipynb`](../notebooks/01_colab_prepare_and_index.ipynb):

1. mount Drive;
2. clone repo hoặc trỏ tới bản đã upload;
3. tải `AIGuruTinix/ViFinQA`;
4. audit;
5. build manifest 6 worker tối đa;
6. build BM25;
7. lưu artefact vào `MyDrive/vifinqa-artifacts`;
8. chạy test.

Nếu runtime reset, copy artefact Drive về local VM trước khi chạy; không index trực tiếp trên Drive vì
nhiều I/O nhỏ chậm và dễ timeout.

## 4. Kaggle GPU

Mở [`02_kaggle_dense_and_generate.ipynb`](../notebooks/02_kaggle_dense_and_generate.ipynb):

1. bật GPU;
2. bật Internet để tải model hoặc attach model checkpoint;
3. attach repo hiện tại, corpus và artefact CPU;
4. xác nhận cell phần cứng;
5. build dense index và copy ra `/kaggle/working/artifacts`;
6. chạy hybrid retrieval;
7. chạy fail-closed retrieval QC và giữ file hash/report;
8. copy hybrid retrieval về artefact store và build diverse annotation pool;
9. khởi động vLLM;
10. chạy generation `--limit 5`;
11. đọc `errors.jsonl` và inspect CSV/query;
12. mới bỏ `--limit`;
13. validate đủ 1.012;
14. package ZIP.

Notebook yêu cầu opt-in để tránh vô tình chạy full/quota:

```bash
export VIFINQA_RUN_FULL=1
```

Với run dùng để nộp, phải thêm `VIFINQA_FINAL_RUN=1` và hai commit SHA phát hành trước cutoff:

```bash
export VIFINQA_FINAL_RUN=1
export VIFINQA_MODEL_REVISION=8e8ed243bbe6f9a5aff549a0924562fc719b2b8a
export VIFINQA_DENSE_REVISION=5617a9f61b028005a4858fdac845db406aefb181
export VIFINQA_TP=1
```

Không dùng chuỗi `main` làm revision. Nếu dense index đính kèm được build bằng revision khác/không
pin, notebook final sẽ dừng. Project cũng phải là Git checkout để lưu `PROJECT_SHA`.

Notebook tự dùng `torch.cuda.device_count()` cho tensor parallel. Nếu model 7B vừa một GPU, TP=1
thường đơn giản hơn; chỉ dùng nhiều GPU sau khi smoke server thành công.

## 5. Pool retrieval sau GPU run

Không gán nhãn chỉ từ BM25 control. Sau khi notebook tạo `retrieval_hybrid.jsonl`, chạy:

```bash
python scripts/61_build_qrels_pool.py \
  --template annotations/qrels_template.jsonl \
  --run bm25=outputs/retrieval.jsonl \
  --run hybrid=/kaggle/working/artifacts/retrieval_hybrid.jsonl \
  --depth 20 \
  --output /kaggle/working/artifacts/qrels_pool_diverse.jsonl
sha256sum /kaggle/working/artifacts/qrels_pool_diverse.jsonl
```

Script lấy provenance/rank từ `bm25`, `dense`, `fused`, `reranked` nếu field tồn tại, union và
deduplicate `table_ref`. Không gọi pool này là qrels trước double annotation/adjudication.

## 6. Cài đặt vLLM

Notebook pin `vllm==0.25.1`, phiên bản PyPI được kiểm ngày 02/08/2026. Vì vLLM pin torch/CUDA
riêng, hãy cài `requirements-gpu.txt` trong runtime mới và restart kernel nếu pip thay torch. Không
ép torch Windows trong `requirements.txt` lên Kaggle.

Nếu server không healthy:

- đọc 4.000 dòng cuối `/kaggle/working/vllm.log`;
- kiểm CUDA compute capability/VRAM;
- giảm `--max-model-len` 8192 → 4096;
- giảm `--gpu-memory-utilization` khi notebook còn model dense trong VRAM;
- giải phóng dense model trước khi serve;
- dùng TP=1 nếu AWQ/tensor parallel gặp lỗi;
- không đổi model sang model >14B.

## 7. Checkpoint và resume

`scripts/50_generate_programs.py` ghi từng success vào `predictions.jsonl`, từng failure có stage,
candidate refs và model revision vào `errors.jsonl`; `run_metadata.json` khóa hash retrieval, manifest,
schema, seed và model. Chạy lại với fingerprint khác trong cùng output sẽ bị từ chối thay vì trộn run.
Không xoá checkpoint khi runtime sắp hết; copy cả thư mục generation ra Kaggle Output/Drive.

Một retry có thể thay output dù temperature 0 do backend/version. Luôn pin:

- repo commit;
- model revision;
- vLLM/transformers/torch version;
- seed;
- prompt/schema hash;
- candidate retrieval file hash.

## 8. Secrets và network

Local vLLM không cần secret; script dùng placeholder `local-vllm`. Nếu endpoint có API key, đặt trong
Kaggle Secrets/env `VLLM_API_KEY`, không ghi vào notebook/output. Competition policy không cho model
đóng, nên không chuyển endpoint sang proprietary API.

## 9. Trước khi tải ZIP xuống

```bash
python scripts/40_validate_submission.py \
  /kaggle/working/artifacts/generation/submission.json \
  --questions "$DATA_ROOT/questions/questions.jsonl" \
  --evidence-root /kaggle/working/artifacts/generation
```

Validator phải báo `valid submission: 1012 questions`. Sau đó package và lưu SHA-256. Chưa upload
nếu table-ref contract chưa được dashboard xác nhận.

## 10. Các giới hạn của notebook hiện tại

- Dense/reranker chưa chạy trong môi trường local này vì không có CUDA.
- Notebook hiện chuẩn bị BGE-M3/RRF/reranker/Qwen, chưa cài SPLADE/ColBERT/PieTa. Các nhánh này chỉ
  được thêm sau khi diverse qrels/dev tồn tại và baseline factorized cho thấy cần thiết; tránh tải
  nhiều model nhưng không có metric để quyết định.
- Structured typed IR đã nối tới compiler/executor; panel coverage và Qwen schema compatibility cần
  được chứng minh bằng smoke/full log trên GPU.
- Notebook không tự submit lên dashboard và không tiêu submission quota.
- Kaggle input path thay đổi theo tên dataset; sửa duy nhất cell path/config, không sửa logic module.
