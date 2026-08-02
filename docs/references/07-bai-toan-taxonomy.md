# Taxonomy bài toán ViFinQA và coverage contract

**Trạng thái:** `verified local` cho thống kê corpus/companion; taxonomy dưới đây là phân tích kỹ
thuật, **không phải nhãn difficulty chính thức** của 1.012 câu hỏi.

## 1. Ranh giới bằng chứng

Release công khai chỉ có câu hỏi và corpus, không có answer, qrels, difficulty hay template ID. Vì
vậy không được suy ra “easy/medium/intermediate/hard” từ thứ tự ID hoặc từ khóa. Các count lexical
như `tổng`, `tỷ lệ`, `cao nhất` có thể chồng lấp và chỉ dùng để stratify audit.

Companion tại commit `9a046de` cung cấp một coverage contract hữu ích:

- bốn tier sinh dữ liệu được mô tả trong README;
- 9 scenario trung gian trong `generation/scenarios.py`;
- 3 công thức trung gian được bật: ROA, ROE, quick ratio;
- 70 typed grammar hard trong `generation/hard/template_intents.py`.

Không có trường nối ngược từng câu release với scenario/template. Do đó grammar gốc chỉ xác định
những năng lực executor/evaluator cần bao phủ, không xác định nhãn cho từng câu.

## 2. Năm trục độc lập

| Trục | Mức | Ví dụ lỗi nếu gộp sai |
|---|---|---|
| evidence scope | một cell; một bảng; nhiều bảng cùng report; nhiều report; nhiều công ty | tìm đúng metric nhưng sai entity/year |
| temporal | instant; duration; hai kỳ; chuỗi thời gian; kỳ kế tiếp | lấy số đầu kỳ như dòng tiền trong kỳ |
| population | entity cụ thể; peer group; cohort lọc; toàn corpus | collapse cohort thành ticker đầu tiên |
| operator | lookup; arithmetic; aggregate; predicate/set; rank; finance formula | program chạy được nhưng sai mẫu số/cohort |
| answerability | đủ dữ liệu; thiếu route; thiếu metric; mâu thuẫn; OCR mơ hồ | model bịa giá trị thay vì abstain |

Độ khó end-to-end là tích của năm trục, không phải chỉ số phép tính. Một lookup qua 100 báo cáo có
thể khó retrieval hơn một ratio trong một bảng.

## 3. Scenario trung gian đã đối chiếu code

| Scenario | Evidence | Operator hợp lệ |
|---|---|---|
| `same_doc_two_inputs` | hai input cùng tài liệu | difference, ratio, share |
| `same_company_two_periods` | một công ty, hai kỳ | difference, growth |
| `two_companies_same_period` | hai công ty, một kỳ | difference |
| `same_doc_multi_inputs` | nhiều input cùng tài liệu | sum, average, min, max, count |
| `same_company_time_series` | chuỗi thời gian | growth, average, min/max, count, argmin/argmax |
| `peer_group_same_period` | nhóm công ty, một kỳ | average, min/max, count |
| `peer_group_two_periods` | nhóm công ty, hai kỳ | growth, average, min/max, count |
| `same_doc_multi_role_formula` | nhiều vai trò metric cùng tài liệu | ratio theo công thức |
| `peer_group_longitudinal` | nhóm công ty nhiều kỳ | growth |

Ba công thức companion hiện thực thi:

- `ROA = net_income / mean(total_assets_begin, total_assets_end)`;
- `ROE = net_income / mean(equity_begin, equity_end)`;
- `quick_ratio = (current_assets - inventory) / current_liabilities`.

Đây là định nghĩa của generator, không mặc định thay thế wording của đề hoặc chuẩn kế toán khác.

## 4. Grammar hard: các họ năng lực

70 template hard có terminal distribution: 32 lookup, 10 maximum, 9 share, 6 count, 6 difference,
5 mean và 2 ratio-of-sums. Terminal đơn giản không có nghĩa reasoning đơn giản: trước terminal có thể
có filter, ratio, growth, set intersection, quantile và entity lookup phụ thuộc.

| Họ grammar | Phép cần có | Bẫy chính |
|---|---|---|
| filter → rank → lookup | threshold, restrict, argmin/max, same-entity lookup | mất identity sau sort/filter |
| two-period transition | growth/difference, sign transition, intersection | nhầm hướng thời gian hoặc denominator |
| cohort statistic | median/quantile/top-n, split cohort, group mean | tie policy, NaN, population leakage |
| share/ratio-of-sums | restricted sum, total sum, scalar share | mean of ratios ≠ ratio of sums |
| financial formula | margin, inventory days, CCC, DOL, debt share | sign convention, average balance, day basis |
| dependent lookup | chọn entity bằng metric A rồi lấy metric B/kỳ kế | retrieval thiếu bảng phụ |
| multi-predicate set | positive/negative/threshold, intersection, count | AND bị biến thành OR |
| sensitivity/scenario | perturb input rồi recompute | thay output trực tiếp thay vì input |

## 5. Ngách bài toán cần bao phủ

1. **Near-duplicate filing retrieval:** cùng biểu mẫu và nhãn xuất hiện ở nhiều company/year.
2. **Row/column/cell selection:** đúng bảng nhưng sai cột kỳ hoặc dòng subtotal.
3. **Continuation table:** bảng nhiều trang, header lặp, phần tổng ở trang sau.
4. **Scope:** hợp nhất/riêng lẻ/không xác định; không tự default.
5. **Restatement:** số so sánh có thể trình bày lại; provenance phải giữ report phát hành nào.
6. **Scale and currency:** đồng/nghìn/triệu/tỷ, VND/USD, ratio/%/percentage point.
7. **Sign semantics:** ngoặc âm, chi phí trình bày âm hoặc dương, dash không tự động là zero.
8. **Instant vs duration:** balance-sheet instant khác flow theo kỳ.
9. **Cohort construction:** industry, fixed threshold, top quantile, survivors qua nhiều kỳ.
10. **Tie and missing policy:** argmax/top-k/median phải định nghĩa tie, NaN, thiếu năm.
11. **Formula ambiguity:** average balance, growth denominator, calendar-day basis.
12. **Unanswerable:** thiếu data, retrieval failure, conflict hoặc parse ambiguity là bốn trạng thái khác.

## 6. Coverage contract cho IR và test

Mỗi operator phải có unit, property-based và differential test. Một run end-to-end chỉ được gọi là
“hard coverage” khi có ít nhất một gold/manual case cho từng họ grammar, không chỉ từng terminal.

| Capability | Unit test | Gold/manual test | Promote gate |
|---|---|---|---|
| lookup/provenance | coordinate round-trip | đúng entity/year/scope | 100% traceable |
| arithmetic/ratio | NumPy/reference | formula + unit đúng | no silent coercion |
| aggregate/rank | ties/NaN/order | cohort membership | exact set agreement |
| temporal | start/end/flow | two-period evidence | ordered periods đúng |
| set/filter | AND/OR/boundary | survivor set | intermediate set exact |
| dependent lookup | identity preserved | second metric/table | route coverage 100% |
| finance formula | formula fixtures | manual accounting review | definition recorded |
| abstention | counterfactual missing fact | adjudicated unanswerable | risk–coverage reported |

## 7. Ưu tiên thực hiện

- `applied`: lookup, scalar arithmetic, aggregate/count/rank baseline, provenance, unit gate.
- `next`: population/cohort selector, dependent lookup, temporal transitions, ratio-of-sums, formula
  registry và intermediate-fact trace.
- `ablation`: iterative subtable selection, finance-card rerank, graph cross-path verification.
- `deferred`: full multimodal OCR/RAG và global GraphRAG vì corpus hiện đã có HTML, câu hỏi chủ yếu
  cần evidence chính xác hơn là query-focused summarization.
