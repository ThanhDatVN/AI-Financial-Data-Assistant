# Báo cáo rà soát và các quyết định đã kiểm chứng

**Mốc rà soát:** 02/08/2026 (Asia/Ho_Chi_Minh)

**Phạm vi:** đặc tả cuộc thi, toàn bộ ViFinQA public, repo companion, tài liệu hiện có, paper,
model card, runtime và mã nguồn trong repository này.

## 1. Kết luận điều hành

Kiến trúc đúng cho bài toán là pipeline có provenance:

`câu hỏi → QuerySpec → route metadata → sparse/dense retrieval → rerank → chọn ô theo schema →
Pandas expression → sandbox execution → validate → submission.zip`.

Ba điều không được đánh đồng:

1. **Thông tin đã xác nhận trong đặc tả**: schema ZIP/JSON, ba nhóm metric, giới hạn model.
2. **Contract của repo companion**: `doc_name|table_N`, F2, tolerance 0,01 trong config paper.
3. **Contract thật của dashboard**: vẫn cần một submission contract-test; ví dụ chính thức lại dùng
   `doc|350`, không giải thích `350`.

Không có gold answer, gold program, gold evidence, split hay difficulty label trong bản public. Vì
vậy không thể báo một accuracy có ý nghĩa chỉ từ release hiện tại. Mọi con số dev/test trước khi có
qrels phải được gắn nhãn proxy hoặc manual audit.

## 2. Những điểm sai/thiếu trong tài liệu cũ

| Chủ đề | Nội dung cũ/rủi ro | Kết quả kiểm chứng và quyết định |
|---|---|---|
| Số bảng | 143.815 được xem là ground truth | Regex cân bằng trên toàn corpus cho **146.246** bảng; README companion hiện cũng ghi 146.246, còn `docs/data.md` và dataset card vẫn ghi 143.815. Dùng 146.246 cho release đã tải, ghi checksum/commit. |
| Difficulty | Suy diễn 5 tầng hoặc khoảng ID | Public release không có nhãn. Companion mô tả **4** tầng: easy, medium, intermediate, hard. Không dùng ID làm nhãn. |
| Table ref | Giả định `doc|offset` hoặc đã giải quyết | Companion commit `9a046de` dùng `doc|table_N`, 1-based trong test/fixture. Dashboard vẫn phải contract-test. Format là config, không hard-code. |
| Scope | Mặc định consolidated | Có 957 consolidated, 954 separate, 7 aggregated, 55 unknown. Chỉ hard-filter khi câu hỏi nói rõ; scope rõ vẫn cho phép tài liệu `unknown`, không tự trộn scope đối nghịch. |
| Đơn vị | Mặc định VND/triệu VND khi thiếu | Cấm mặc định. Parser giữ `UNKNOWN`; answer chỉ sinh khi unit lineage được chứng minh. |
| Vai trò LLM | “LLM không đọc số” tuyệt đối | Quy tắc đúng: LLM không chép **giá trị ô nguồn**; vẫn được đọc năm/ngưỡng trong câu hỏi và schema/label. Giá trị nguồn chỉ đi qua parser/executor. |
| Model 14B | Qwen2.5-Coder-14B được xem là hợp lệ | Model card ghi 14,7B; loại khỏi cấu hình nghiêm ngặt `≤14B`. Chọn Qwen2.5-Coder-7B-Instruct-AWQ (base 7,61B). |
| Answer tolerance | 0,01 là luật BTC | 0,01 xuất hiện trong config/evaluator companion; đặc tả public chỉ nói “trong ngưỡng sai số”. Gắn nhãn paper-parity, chờ dashboard xác nhận. |
| Kaggle/Colab | Cam kết GPU/quota cố định | Phần cứng và hạn mức biến động. Notebook phát hiện runtime, checkpoint và không giả định T4/P100 hay số giờ cố định. |
| vLLM structured output | Dùng API cũ `GuidedDecodingParams` | Tài liệu vLLM hiện dùng structured outputs/JSON schema; pipeline gọi OpenAI-compatible `response_format=json_schema`. |
| TT200/2014 | Được mô tả như quy định còn hiệu lực | Thông tư 200 có giá trị lịch sử với corpus 2015–2025 nhưng đã hết hiệu lực từ 01/01/2026 khi TT99/2025 thay thế. |

## 3. Audit dữ liệu tái lập

Lệnh:

```powershell
.venv\Scripts\python.exe scripts\00_audit_dataset.py
.venv\Scripts\python.exe scripts\10_build_manifest.py --workers 6
```

Kết quả từ `data/interim/dataset_audit.json` và manifest đầy đủ:

| Chỉ số | Kết quả |
|---|---:|
| Câu hỏi | 1.012, ID duy nhất và liên tiếp 1–1.012 |
| Báo cáo | 1.973 |
| Doanh nghiệp | 100 ticker |
| Năm | 2015–2025 |
| Kích thước text | 380.236.765 byte = 362,622 MiB |
| Bảng HTML cân bằng | 146.246 |
| `table_ref` duy nhất trong manifest | 146.246 |
| Tài liệu có bảng | 1.965 |
| Tài liệu không bảng | 8, đều là thư giải trình PRT |
| Bảng/tài liệu | min 0; mean 74,124; median 70; p95 116; max 248 |
| Trang/tài liệu | median 57; p95 97; max 206 |
| Lỗi cặp `<table>`/`</table>` | 0 |
| Bảng rỗng/lệch chiều sau parse toàn bộ | 0/0 |

Phân bố scope báo cáo: consolidated 957; separate 954; aggregated 7; unknown 55. Phân bố
năm lần lượt 125, 146, 164, 173, 180, 196, 193, 201, 195, 200, 200.

Sau hiệu chỉnh NLU trên cả 1.012 câu, target unit không còn `UNKNOWN`: % 260; tỷ đồng 241;
triệu đồng 217; nghìn tỷ đồng 77; trăm tỷ đồng 64; year 51; ratio 38; shares 24;
nghìn đồng 16; count 12; VND 6; triệu cổ phiếu 5; triệu USD 1. Company resolver chỉ để
trống câu 464 vì câu đó cố ý nói toàn bộ công ty mà không nêu ticker.

Sau khi sửa lỗi context của bảng trước làm rò `%`/VND sang bảng sau, unit mặc định trên 146.246 bảng
là: VND 74.133; triệu VND 29.584; unknown 29.068; % 7.902; cổ phiếu 3.744; nghìn VND 942;
USD 509; tỷ VND 349; triệu USD 15. `UNKNOWN` tăng có chủ ý vì parser fail-closed. Evidence còn
resolve unit theo `column_label` rồi `row_label`, nên bảng hỗn hợp có thể giữ table unit `UNKNOWN`
nhưng ô USD vẫn có lineage USD; không tự ép bảng sang VND.

## 4. Bằng chứng chạy thực tế

- Full parser/manifest sau sửa context/unit: 146.246 bảng, 6 worker, 303,0 giây; JSONL 426.422.713
  byte, Parquet 50.139.191 byte; Parquet SHA-256 `060bd26e...69e29d`.
- Full BM25 rebuild: 146.246 bảng, 31,4 giây; index 170.777.097 byte trong `data/index/bm25/`.
- Retrieval toàn bộ 1.012 câu: lọc metadata trước ranking và round-robin theo `ticker × report_year`;
  0 câu rỗng, 0 ref sai route, 0 route khả dụng bị thiếu. Ba mốc không có report riêng dùng report
  năm kế tiếp chứa cột so sánh: PDR-2024, BSR-2016, BID-2021.
- Integration câu 1: top-1 là `VJC_financial_statements_2018_separate|table_50`; ô r0/c1 chứa
  lãi tiền gửi 2018; executor tính `208253.201298` triệu đồng từ CSV evidence.
- Integration câu 213: bảng ACV hỗn hợp giữ table unit `UNKNOWN`, ô r0/c1 nhận USD và tính
  `6.15569834` triệu USD.
- Retrieval mới mất 54,9 giây, SHA-256 `99bcd364...38bf09`; QC machine-readable tại
  `outputs/retrieval_qc.json` xác nhận 0 failure và 289 câu multi-entity.
- Test suite hiện có 56 test unit/integration; Ruff/format xanh và mypy strict trên 53 file nguồn.
- Template qrels schema v2 có 100 câu, SHA-256 `1584fe25...a0114`, 100% `unlabeled` và không chứa
  gold giả. Pool tooling đã tạo 1.998 candidate từ BM25 retrieval v2, SHA-256 `8d3519be...ac4eba`;
  chưa phải diverse
  pool và chưa phải accuracy evidence.

Artefact corpus/index/output được gitignore vì lớn. Metadata, code và lệnh tái sinh được commit.

## 5. Các blocker không thể tự suy diễn

1. Dashboard có chấp nhận `doc|table_N` hay yêu cầu `doc|N`/offset khác?
2. Tolerance chính thức và quy tắc rounding là gì?
3. Gold/qrels hoặc dev feedback nào được BTC cung cấp trên dashboard?
4. Model cutoff được xét theo ngày model card, revision hay ngày weight công bố?

Các blocker này không ngăn phát triển. Chúng là gate trước submission chính thức, cần lưu ảnh/FAQ
hoặc kết quả contract-test trong nhật ký thí nghiệm.
