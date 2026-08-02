# 03 — Financial table QA và suy luận số có căn cứ

## 1. Benchmark nền tảng đã kiểm chứng

| Paper | Bằng chứng liên quan | Áp dụng |
|---|---|---|
| [FinQA](https://arxiv.org/abs/2109.00122), Chen et al. 2021 | QA tài chính với reasoning program và multi-step numerical reasoning | toán hạng phải neo vào evidence, lưu execution trace |
| [TAT-QA](https://arxiv.org/abs/2105.07624), Zhu et al. 2021 | kết hợp table/text và operator số học/đếm/so sánh/sort | operator inventory và hybrid evidence |
| [DataFrame QA](https://arxiv.org/abs/2401.15463), Ye et al. 2024 | DataFrame/Pandas reasoning | execution-based validation |
| [RePanda](https://arxiv.org/abs/2503.11921), Chegini et al. 2025 | Pandas-powered verification/reasoning | biểu thức có thể chạy lại thay natural-language answer |

FinQA/TAT-QA không phải train/dev thay thế cho ViFinQA: khác ngôn ngữ, schema, corpus và output
contract. Chúng được dùng để thiết kế operator/trace, không dùng số leaderboard của paper để ước lượng.

## 2. Bản chất bài toán ViFinQA

Đây không chỉ là QA trên bảng mà là chuỗi ràng buộc đồng thời:

1. đúng entity, năm báo cáo và scope;
2. đúng bảng/cell nguồn và table-ref contract;
3. parse đúng locale/OCR và đơn vị nguồn;
4. đúng operator, chiều phép tính và denominator;
5. answer float đúng target unit;
6. Pandas chạy lại trên CSV package và cho cùng answer.

Một answer đúng do vô tình chọn nhầm evidence vẫn là pipeline không đáng tin; một query chạy được nhưng
sai unit cũng không đạt. Vì vậy cần đánh giá retrieval, grounding, program, execution và unit riêng.

## 3. Failure taxonomy dùng cho kiểm thử

| Lỗi | Dấu hiệu | Gate |
|---|---|---|
| entity/scope drift | đúng chỉ tiêu nhưng sai ticker/consolidated/separate | QuerySpec + route coverage |
| temporal drift | nhầm năm báo cáo, kỳ đầu/cuối, prior period | temporal regression fixtures |
| evidence omission | thiếu một ticker/năm trong aggregate | operand coverage matrix |
| layout confusion | nhầm header/rowspan/continuation table | coordinate + parser fixtures |
| numerical drift | LLM chép/biến đổi số | không đưa source digits vào prompt |
| unit drift | nghìn/triệu/tỷ/%, shares | dimensional conversion + magnitude checks |
| operator drift | diff/growth/ratio đảo chiều | typed operator invariants |
| accidental execution | biểu thức chạy nhưng chọn sai row/column | gold-cell tests + trace |

## 4. Operator inventory mục tiêu

```text
lookup, add, subtract, multiply, divide,
sum, mean, min, max, count_if,
diff, growth_pct, ratio, argmin, argmax
```

Mỗi operand cần `(table_ref, row_index, column_index, source_unit)`. `target_unit` nằm ngoài operand và
compiler thực hiện conversion một lần. Phép cộng/trừ yêu cầu dimension tương thích; `growth_pct`
phải ghi convention `(end-start)/start*100`; division-by-zero không được che lỗi.

## 5. Text evidence

TAT-QA nhắc rằng câu trả lời tài chính đôi khi cần text và table. Release ViFinQA chủ yếu yêu cầu
table evidence theo submission schema; context ngoài bảng vẫn hữu ích cho title/unit/scope nhưng không
được biến thành số CSV nếu số đó không truy vết được tới cell/table theo contract. Cần hỏi BTC nếu câu
gold phụ thuộc con số ngoài `<table>`.

## 6. Dev data không có sẵn

Không gọi 1.012 câu hỏi public là train/dev và không tune trực tiếp qua leaderboard quá mức. Bộ fixture
nội bộ cần:

- một tập gold-table/gold-cell thủ công, versioned và review chéo;
- synthetic cases sinh từ cell corpus cho parser/operator regression, tách khỏi evaluation chính;
- perturbation đổi dấu, dấu phân cách nghìn, dash/ngoặc âm, alias và header;
- multi-entity coverage cases;
- clean-room package execution.

Synthetic data có thể kiểm code nhưng không chứng minh generalization trên câu hỏi thật.

## 7. Trạng thái

Scalar/aggregate/count-if/arg-extremum IR, dimension checker, executor và q1 real-corpus integration đã
có. Structured IR/schema/grounding và basic ticker-year coverage đã có code/test, nhưng chưa có bằng
chứng Qwen GPU; semantic hard-panel, gold dev set và full Qwen run vẫn TODO/PARTIAL.
