# AI Financial Data Assistant — ViFinQA

Pipeline truy hồi bảng và sinh truy vấn Pandas có căn cứ cho 1.012 câu hỏi tài chính tiếng Việt trên
1.973 báo cáo ViFinQA (100 mã chứng khoán, 2015–2025).

## Trạng thái kiểm chứng ngày 02/08/2026

- Đã tải/audit toàn bộ release: **146.246 raw tables**, không phải 143.815 như một số metadata cũ.
- Đã build full manifest và BM25; 146.246 table-ref duy nhất, 0 bảng empty/malformed.
- Đã route/retrieve 1.012 câu bằng CPU; không có candidate rỗng, ref sai metadata hay route khả dụng
  bị thiếu; validator fail-closed lưu hash manifest/retrieval. Đây là health check, không phải Recall
  vì release không có qrels/gold.
- Đã chặn rò unit/context từ bảng trước, thêm unit USD theo ô cho bảng hỗn hợp và regression thực trên
  câu 213; manifest/BM25/retrieval đã rebuild sau thay đổi.
- Đã có parser, QuerySpec, sparse/dense interfaces, evidence CSV, typed scalar/aggregate/rank IR,
  structured IR parser, dimension/grounding checker, deterministic compiler, isolated executor,
  validator/packager và notebook Colab/Kaggle.
- Đã nâng template 100 qrels lên schema v2 (answerability/intermediate facts/cohort/formula) và tạo
  pool control 1.998 candidate từ frozen retrieval v2; toàn bộ vẫn `unlabeled`, chưa dùng để báo metric.
- GPU dense/reranker/Qwen full run, panel hard-tier coverage và dashboard contract vẫn phải hoàn tất; không
  được ghi là “passed” trước khi có artefact thật.

## Quy định đang được áp dụng

Đặc tả local cấm model đóng, chỉ cho model công khai phát hành trước 01/06/2026 và ≤14B. Baseline dùng
Qwen2.5 7B AWQ; các model 14,7–14,8B bị loại. Mọi model dùng trong run cuối phải pin revision SHA và
ghi license/environment. Không gọi API proprietary ở bất kỳ khâu nào.

Hai điểm chỉ dashboard/BTC có thể chốt — format chính xác của `relevant_tables` và answer tolerance —
được giữ là `unresolved`. Companion commit `9a046de` dùng `doc|table_N` và tolerance 0,01 cho paper
parity, nhưng đây chưa được coi là contract dashboard.

Lịch 01/08–06/09/2026, quota nộp và yêu cầu ZIP trong [đặc tả local](cuoc-thi-text-to-pandas-bctc.md)
phải được đối chiếu Dashboard trước mỗi submission vì policy có thể cập nhật.

## Tài liệu chính

| Tài liệu | Nội dung |
|---|---|
| [Báo cáo rà soát](docs/00-bao-cao-ra-soat.md) | Sai lệch đã phát hiện và cách sửa |
| [Phân tích đề bài](docs/01-phan-tich-de-bai.md) | Contract, metric, audit dữ liệu và câu hỏi mở |
| [Phân tích vấn đề](docs/02-phan-tich-van-de.md) | Failure modes và biện pháp kiểm soát |
| [Kế hoạch chi tiết](docs/03-ke-hoach-thuc-hien.md) | Milestone, gate, artefact và trạng thái thật |
| [Colab/Kaggle](docs/04-huong-dan-kaggle-colab.md) | Hướng dẫn hosted runtime, checkpoint, validate |
| [Runbook](docs/05-runbook.md) | Lệnh tái lập local từ audit đến ZIP |
| [Báo cáo kiểm thử](docs/06-bao-cao-kiem-thu.md) | Test đã chạy và mục chưa có bằng chứng |
| [Working-notes outline](docs/07-working-notes-paper-outline.md) | Khung paper không cho claim thiếu log |
| [Qrels nội bộ](docs/08-huong-dan-gan-nhan-qrels.md) | Sampling, gán nhãn kép và adjudication |
| [Final-run preflight](docs/09-final-run-preflight.md) | Local evidence, pinned revisions và Kaggle handoff |
| [Chạy thử nghiệm GPU từng bước](docs/10-huong-dan-chay-thu-nghiem.md) | Upload input, smoke, full run, checkpoint và xử lý lỗi |
| [Deep research](docs/references/README.md) | Ma trận paper/model/policy và BibTeX đã kiểm |
| [Nhật ký thí nghiệm](docs/nhat-ky-thi-nghiem.md) | Run/submission log bắt buộc |

## Thiết lập local

Yêu cầu Python 3.12. Trên PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps --no-build-isolation
.\.venv\Scripts\python.exe -m pytest
```

Các lệnh audit/build/search/validate đầy đủ ở [runbook](docs/05-runbook.md). Dữ liệu/index/output lớn
được gitignore; chúng có thể tái sinh từ release và scripts.

## Notebook GPU

- [Colab prepare/index](notebooks/01_colab_prepare_and_index.ipynb): tải/audit/build manifest + BM25.
- [Kaggle dense/generate](notebooks/02_kaggle_dense_and_generate.ipynb): BGE-M3/FAISS, hybrid retrieval,
  Qwen2.5-Coder-7B-Instruct-AWQ qua vLLM, checkpoint, validate và package.

Hosted GPU/quota không được giả định cố định. Notebook tự kiểm phần cứng, chạy smoke trước full và
không tự submit lên dashboard.

## License và nguồn

ViFinQA được phát hành CC BY-NC 4.0. Hiện pipeline không dùng dữ liệu số tài chính bên ngoài. Nguồn
nghiên cứu, model card và văn bản kế toán tham chiếu được kê khai tại
[docs/references](docs/references/README.md).
