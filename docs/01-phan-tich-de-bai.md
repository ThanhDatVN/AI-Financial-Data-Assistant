# Phân tích đề bài và contract cuộc thi

**Nguồn chuẩn nội bộ:** [`cuoc-thi-text-to-pandas-bctc.md`](../cuoc-thi-text-to-pandas-bctc.md)

**Nguyên tắc:** khi tài liệu này khác đặc tả hoặc dashboard, đặc tả/dashboard được ưu tiên. Mọi
điểm chưa rõ phải được contract-test, không “điền vào chỗ trống” bằng giả định.

## 1. Bài toán cần giải

Với mỗi câu hỏi tài chính tiếng Việt và kho BCTC OCR, hệ thống phải đồng thời tạo:

1. danh sách báo cáo liên quan (`relevant_docs`);
2. danh sách bảng chứa toàn bộ hoặc một phần số liệu cần tính (`relevant_tables`);
3. CSV evidence nằm trong `data/` của gói nộp;
4. một `pandas_query` chạy lại được trên các DataFrame evidence;
5. một `answer` số thực đúng bằng kết quả thực thi query trong tolerance chấm bài.

Đây không chỉ là QA. Nó là bài toán kết hợp corpus routing, table retrieval, hiểu kỳ kế toán,
chuẩn hoá số liệu, program synthesis và reproducible execution.

## 2. Contract đầu vào đã kiểm chứng

Public release `AIGuruTinix/ViFinQA` đã audit gồm 1.012 câu hỏi, 1.973 báo cáo, 100 ticker,
giai đoạn 2015–2025 và 146.246 bảng HTML thô cân bằng. Báo cáo là UTF-8 text có marker trang
và inline `<table>`. Public release **không** có:

- answer;
- pandas program;
- gold report/table;
- normalized table CSV;
- official train/dev/test split;
- difficulty label.

Tên split `train` trên Hugging Face chỉ là quy ước đóng gói. Không được dùng nó như bằng chứng rằng
1.012 câu là training set có label.

## 3. Contract đầu ra

Submission là ZIP có đúng một JSON ở root và thư mục `data/`:

```text
submission.zip
├── submission.json
└── data/
    ├── q1_df1.csv
    └── ...
```

Mỗi phần tử JSON:

```json
{
  "id": 1,
  "question": "...",
  "answer": 208253.201298,
  "relevant_docs": ["VJC_financial_statements_2018_separate"],
  "relevant_tables": ["VJC_financial_statements_2018_separate|table_50"],
  "evidence": [
    {"variable": "df1", "csv_path": "data/q1_df1.csv"}
  ],
  "pandas_query": "float(df1.loc[(df1['row_index']==0) & (df1['column_index']==1), 'base_value'].iloc[0]) / 1000000.0"
}
```

Quality gate trong repository kiểm:

- đủ và chỉ đúng 1.012 ID, không trùng;
- question khớp byte-level với source;
- answer hữu hạn;
- docs/tables không rỗng, không trùng, table thuộc doc đã khai báo;
- variable là Python identifier và duy nhất trong câu;
- `csv_path` là POSIX relative path bắt đầu `data/`, không traversal;
- CSV tồn tại;
- AST của query không có import/dunder/tên lạ;
- kết quả thực thi query khớp `answer`.

## 4. Điểm chưa rõ trong `relevant_tables`

Đặc tả đưa ví dụ `AAA...|350` nhưng không định nghĩa `350` là ordinal, dòng, trang hay offset. Repo
companion tại commit `9a046de2f2daea4d2be0a05d4a5f3f1220e6922a` dùng canonical
`doc_name|table_N`; fixture/test bắt đầu từ `table_1`.

Quyết định triển khai:

- internal identity: `doc_id`, 1-based `table_id`, page, line, char offset, hash HTML;
- default export: `{doc_id}|table_{table_id}` để tương thích companion;
- format và first ID là config;
- metadata ghi `dashboard_verified=false` cho tới contract-test;
- không dùng lượt private để dò format;
- chỉ test public theo kế hoạch nhỏ, ghi lại submission hash và kết quả.

## 5. Cách tính điểm và hệ quả

Đặc tả có ba nhóm tự động:

1. **Retrieval:** precision, recall và F2 trên report/table evidence.
2. **Answer Accuracy:** answer khớp gold trong tolerance.
3. **Execution Accuracy:** pandas query chạy được và trả kết quả đúng.

Với `P` và `R`, `F2 = 5PR / (4P + R)`. Recall được ưu tiên nhưng nộp bảng thừa vẫn làm giảm
precision. Không có một `k` tối ưu cố định cho mọi câu:

- câu một thực thể/một kỳ thường cần ít bảng;
- câu so sánh nhiều ticker/kỳ cần ít nhất một route cho mỗi ticker/kỳ;
- số bảng cuối phải tune bằng qrels/dev feedback, không chọn theo cảm giác;
- retrieval top-k nội bộ có thể lớn, `relevant_tables` nộp phải là tập evidence tối thiểu thực sự
  được query sử dụng.

Answer Accuracy và Execution Accuracy không tự động giống nhau. Chúng chỉ đồng bộ khi `answer`
luôn lấy từ kết quả thực thi cuối cùng, không từ text model hoặc biến trung gian.

## 6. Quy định model và compliance gate

Theo đặc tả hiện có:

- tổng số tham số của model phải `≤14B`;
- model/weight phải phát hành trước 01/06/2026;
- không dùng model đóng/proprietary;
- bài cuối cần working notes paper để kết quả được công nhận;
- private phase tối đa 5 submission tổng cộng; public tối đa 10/ngày theo tài liệu đã cung cấp.

Model mặc định:

| Vai trò | Model | Tham số | License | Quyết định |
|---|---|---:|---|---|
| embedding | BAAI/bge-m3 | ~0,6B | MIT | dùng |
| reranker | BAAI/bge-reranker-v2-m3 | ~0,6B | Apache-2.0 | dùng |
| NLU | Qwen2.5-7B-Instruct-AWQ | 7,61B | Apache-2.0 | dùng/ablation |
| program | Qwen2.5-Coder-7B-Instruct-AWQ | 7,61B | Apache-2.0 | dùng |
| coder 14B | Qwen2.5-Coder-14B | 14,7B theo card | Apache-2.0 | **loại** trong policy nghiêm ngặt |

Trước submission phải lưu model ID, immutable revision, license, parameter count và ngày phát hành.
Không dùng API proprietary kể cả chỉ để rerank hoặc sửa lỗi.

## 7. Hiểu kỳ kế toán

Các mẫu cần phân biệt:

- “trong năm N”: flow, ưu tiên báo cáo N/cột hiện tại;
- “cuối năm N”: point-in-time, ưu tiên báo cáo N/cột hiện tại;
- “đầu năm N”: thường là prior-period column trong báo cáo N;
- “từ N sang N+1”: cần cả hai kỳ;
- “đầu năm đến cuối năm N”: cùng một báo cáo nhưng hai column role;
- fiscal year end lệch 31/12: chỉ dùng khi text báo cáo chứng minh.

Không mặc định lấy báo cáo N+1 cho cuối năm N. Có thể dùng nó như fallback evidence nhưng cần
provenance và đánh giá gold table vì cùng giá trị có thể xuất hiện ở nhiều báo cáo.

## 8. Unit contract

Pipeline giữ ba tầng:

1. `raw_value`: chuỗi OCR nguyên gốc;
2. `numeric_value`: số sau parse theo locale trong đơn vị nguồn;
3. `base_value`: số sau nhân source multiplier khi unit lineage đã biết.

Target divisor đến từ câu hỏi. Ví dụ nguồn VND, target triệu VND: `answer = base_value / 1e6`.
Không đổi % thành fraction nếu đề hỏi phần trăm; không giả định dash luôn bằng 0 nếu ngữ cảnh không
cho phép; không parse OCR `O/0` mơ hồ thành số.

## 9. Nguyên tắc evidence-grounded

- Model thấy question, metadata, header, row label và coordinate; model không cần thấy giá trị ô để
  chép số.
- Mọi số nguồn được parser lấy từ CSV và executor sử dụng.
- Model được đọc số nằm trong **câu hỏi** (năm, threshold, top-n) vì đó là logic, không phải source.
- Query được lưu nguyên văn và chạy lại trên đúng CSV đã nộp.
- CSV long giữ `table_ref`, ticker, report year, scope, row/column index, raw/numeric/base value,
  source unit và label.

## 10. Lịch cuộc thi trong tài liệu nguồn

Tài liệu đã cung cấp ghi public 01/08–31/08/2026, private 01/09–03/09/2026, công bố 06/09/2026.
Search công khai không tìm được trang crawlable xác nhận lịch/dashboard; vì đây là thông tin biến
động, người vận hành phải đối chiếu trực tiếp dashboard trước mỗi gate. Không xem bảng lịch này là
thay thế thông báo BTC.

## 11. Definition of done cho một prediction

Một câu chỉ được gắn `ready` khi:

1. route bao phủ mọi entity/kỳ cần thiết;
2. bảng được chọn chứa đúng label/column hoặc đủ dữ liệu cho phép tính;
3. unit lineage không mơ hồ;
4. pandas query qua AST policy và chạy thành công;
5. answer lấy trực tiếp từ executor;
6. evidence CSV tái hiện đúng query;
7. table/doc refs đúng contract đang active;
8. trace có config hash, model revision, table hash và lỗi/warning.
