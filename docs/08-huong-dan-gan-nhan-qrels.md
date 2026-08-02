# Hướng dẫn tạo qrels nội bộ

Mục tiêu là tạo tập chẩn đoán retrieval/program, không dò leaderboard. Template được lấy mẫu từ public
questions nhưng chỉ trở thành gold sau gán nhãn và review.

## Quy trình

1. Chạy `scripts/60_sample_qrels.py --size 100` với seed mặc định.
2. Chạy retrieval systems đã khóa rồi pool bằng `scripts/61_build_qrels_pool.py`; union tối thiểu
   BM25 original/accentless, dense, late-interaction hoặc reranker và exact metadata route.
3. Annotator A và B gán pool độc lập: `not_relevant`/`partial`/`relevant` và
   `insufficient`/`sufficient`; mỗi người giữ bản riêng cho tới khi hoàn tất.
4. Hai annotator điền/đối chiếu docs, tables, cells, intermediate facts, cohort, operator, formula,
   unit, answer và answerability state.
5. Adjudicator giải quyết bất đồng; chỉ `status=adjudicated` mới được tính metric.
6. Freeze file, pool, schema, SHA-256 và grouped split trước ablation.

Sampling round-robin theo entity count (0/1/multi), year count (0/1/multi), scope và target unit để
không bị 100 câu lookup đơn giản chi phối.

Template hiện tại có 100 ID duy nhất trên 77 strata, SHA-256
`1584fe25a4ba7889d822f93ecaa8916e00fe57f01700172f09ffbda7136a0114`. Schema v2 thêm
fingerprint split, answerability, intermediate facts, cohort, operator/formula, unit/tolerance/rounding
và ba vai trò review; tất cả dòng vẫn `status=unlabeled` và các trường gold rỗng.

Pool control hiện có 100 câu/1.998 candidate từ frozen BM25 retrieval v2, SHA-256
`8d3519bec6dcc6baf08ba4b1d01f8e445e72c0140c2f59140fe2544049ac4eba`. Nó chỉ kiểm tooling;
không bắt đầu annotation cuối cho tới khi bổ sung các run đa dạng để giảm pool bias.

Ví dụ:

```powershell
.venv\Scripts\python.exe scripts\61_build_qrels_pool.py `
  --run bm25=outputs/retrieval.jsonl `
  --run hybrid=outputs/retrieval_hybrid.jsonl `
  --run late=outputs/retrieval_late.jsonl `
  --depth 20
```

## Quy tắc evidence

- `gold_tables` chỉ chứa bảng có cell cần cho phép tính, không thêm bảng “liên quan chung”.
- `gold_cells` ghi `table_ref`, row/column index, raw label và vai trò operand.
- Nếu năm đích nằm ở cột so sánh của report năm kế tiếp, ghi rõ trong `notes`.
- Không gán VND khi source unit chưa chứng minh.
- Câu không thể trả lời dùng `answerability=MISSING_IN_CORPUS` hoặc trạng thái cụ thể khác, kèm
  `answerability_reason`; lifecycle `status` vẫn là annotated/reviewed/adjudicated. Không điền answer 0.
- `intermediate_facts` phải đủ để kiểm cohort/filter/rank/formula, không chỉ final operands.
- `cohort_members` ghi exact set sau filter; tie/missing policy ghi trong program/notes.
- Table-ref vẫn lưu internal `doc|table_N` và field dashboard verification riêng.

## Kiểm chất lượng

- hai annotator độc lập trên toàn sampling frame mục tiêu; nếu nguồn lực buộc dùng overlap, tối thiểu
  20% stratified và công bố giới hạn;
- kiểm trùng ID/ref, ref tồn tại manifest, cell tồn tại bảng;
- program phải chạy trên evidence CSV và khớp answer;
- kiểm intermediate set/fact và dimension/unit, không chỉ execution success;
- reviewer không được sửa trực tiếp answer mà không để audit note;
- manual-qrels score luôn mang nhãn `internal`, không gọi official/dev leaderboard.

Schema máy đọc nằm ở `configs/qrels.schema.json` và `configs/qrels_pool.schema.json`.
