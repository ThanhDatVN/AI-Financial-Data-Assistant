# Ngữ nghĩa tài chính, thời gian, scope và đơn vị

## 1. Fact model thay vì cell trần

[XBRL Essentials](https://specifications.xbrl.org/xbrl-essentials.html) mô hình hóa fact bằng concept,
entity, period và unit, kèm dimensions. Đây là khung tốt cho internal evidence dù corpus không phải
XBRL. Mỗi selected cell nên thành:

```text
Fact = value + concept + entity + period + scope/dimensions
       + unit + scale + sign + source coordinate + report revision
```

Thiếu một trường quan trọng thì không được silently coerce. `raw_value`, `numeric_value` và
`base_value` phải cùng tồn tại để truy vết.

## 2. Taxonomy metric có version

[IFRS Accounting Taxonomy 2025](https://www.ifrs.org/issued-standards/ifrs-taxonomy/ifrs-accounting-taxonomy-2025/)
được IFRS Foundation xác nhận vẫn current cho reporting 2026. Taxonomy, labels và formula linkbase là
nguồn tham khảo semantic; không được dùng để thay số liệu hoặc áp công thức IFRS lên mọi báo cáo Việt
Nam một cách máy móc.

Metric registry nội bộ cần:

- canonical ID + Vietnamese/English aliases;
- statement family và industry applicability;
- instant/duration type;
- allowed unit/dimension;
- debit/credit hoặc display-sign convention nếu biết;
- formula dependencies và version/effective period;
- scope compatibility và exclusions;
- source/rationale cho từng alias/formula.

Pháp lý Việt Nam phải time-aware: TT200/2014 có ý nghĩa cho corpus lịch sử nhưng được TT99/2025 thay
từ 2026; các biểu mẫu ngân hàng/chứng khoán có quy định riêng. Chỉ dùng văn bản để giải nghĩa schema,
không dùng làm external numeric evidence.

## 3. Temporal semantics

| Loại | Ví dụ | Rule |
|---|---|---|
| instant | tiền, tài sản, nợ tại ngày chốt | chọn đúng date endpoint |
| duration | doanh thu, lợi nhuận, dòng tiền trong kỳ | cần start/end period |
| average balance | ROA/ROE, turnover | mean begin/end theo definition |
| two-period change | chênh lệch/growth | khóa old/new direction |
| next-period dependent | entity chọn ở năm t rồi lookup t+1 | preserve identity và route phụ |
| restated comparative | số năm trước trình bày trong report mới | giữ report revision/provenance |

Date/year phải parse deterministic. Không giao cho tokenizer/LLM tự quyết các dạng `31/12/2023`,
`2023-12-31`, `năm tài chính kết thúc...` nếu rule có thể viết rõ.

## 4. Unit, scale và dimension

1. Normalize Unicode/space nhưng giữ raw.
2. Parse sign: parentheses, leading/trailing minus; không suy dash là zero.
3. Resolve cell override trước table/report default.
4. Convert source scale sang base value bằng exact decimal/rational nếu có thể.
5. Type-check operator trước execution.
6. Convert/round output đúng một lần cuối.

Các dimension tối thiểu: currency, count, days, pure ratio, percent, percentage-point, currency/time và
unknown. Cộng/trừ chỉ hợp dimension; ratio của hai currency cùng basis cho pure ratio; percent change
khác percentage-point difference. Không cộng số hợp nhất với riêng lẻ hoặc VND với USD khi không có
explicit conversion evidence.

## 5. Formula registry

Mỗi formula là object versioned:

```text
id, aliases, inputs(role, concept, temporal_type), expression,
output_dimension, sign_policy, zero_policy, scope_policy,
effective_period, source, tests
```

Các nhóm cần triển khai theo grammar local:

- ROA/ROE/quick ratio;
- margin và asset turnover;
- inventory days và cash-conversion cycle;
- net working capital;
- debt maturity/debt share;
- degree of operating leverage;
- growth/difference/absolute change;
- share, ratio-of-sums và cohort averages.

Công thức trong câu hỏi hoặc companion có ưu tiên hơn công thức mặc định. Nếu hai định nghĩa hợp lý
nhưng đề không phân biệt, state là ambiguous/abstain thay vì chọn tùy ý.

## 6. Logical consistency và cross-check

[SEC-FinTables](https://aclanthology.org/2026.findings-acl.764/) nhấn mạnh lỗi totals/components ở bảng
tài chính. Có thể thêm non-authoritative consistency signals:

- total ≈ sum components trong tolerance scale-aware;
- assets ≈ liabilities + equity khi cùng scope/date;
- subtotal/percentage identity;
- same fact ở table khác/report comparative;
- formula cross-path agreement.

Signal mismatch không được tự “sửa” raw OCR. Nó hạ confidence, mở route bổ sung hoặc gửi manual review.

## 7. Provenance tối thiểu trong answer trace

Mỗi operand ghi `doc_id`, `table_ref`, row/column coordinate, raw/header/context hash, entity, period,
scope, source unit/scale, normalized value và transform. Mỗi operator ghi definition/version. Final
answer ghi target unit, rounding và executor version. Đây là điều kiện để tái lập và adjudicate.
