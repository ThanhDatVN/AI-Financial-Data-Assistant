# 01 — Text-to-Pandas và sinh chương trình thực thi được

## 1. Nguồn gần bài toán nhất

| Nguồn gốc | Bằng chứng liên quan | Chuyển thành quyết định |
|---|---|---|
| [DataFrame QA](https://arxiv.org/abs/2401.15463), Ye et al. 2024 | đặt LLM trên DataFrame QA và sinh Pandas mà không cần phơi toàn bộ dữ liệu | prompt schema/label/coordinate; số nguồn do code đọc |
| [RePanda](https://arxiv.org/abs/2503.11921), Chegini et al. 2025 | dùng Pandas như lớp reasoning/verification có thể thực thi | execute và kiểm answer thay vì tin completion |
| [Text-to-Pipeline](https://arxiv.org/abs/2505.15874), Ge et al. 2025 | predict–execute nhiều bước với feedback cho data pipeline | tách QuerySpec → retrieval → program → executor |
| [DS-1000](https://arxiv.org/abs/2211.11501), Lai et al. 2022 | code data-science cần functional tests và constraint checks | test thực thi + AST allowlist + metamorphic tests |
| [API-assisted Table QA](https://arxiv.org/abs/2310.14687) | API/table representation giúp xử lý cấu trúc bảng khác nhau | long-form evidence schema cố định |

Các paper này không dùng corpus ViFinQA, vì thế benchmark score của chúng không được dùng làm dự báo
Execution Accuracy. Chúng chỉ chứng minh các pattern kiến trúc đáng kiểm thử.

## 2. Điều chuyển được từ Text-to-SQL

Text-to-SQL và Text-to-Pandas cùng có ba nút lỗi: schema linking, sinh logic và kiểm thực thi. Phần
chuyển được là:

- thu hẹp schema trước generation;
- phân rã câu phức thành operator/operand typed;
- sinh nhiều candidate chỉ khi confidence thấp;
- chọn bằng execution và invariants, không bằng độ giống chuỗi;
- lưu trace để tách syntax error, wrong cell, wrong operation và wrong unit.

Không chuyển nguyên SQL grammar, database probing hoặc multi-agent framework: submission cần Pandas
trên CSV do đội xuất, compute/quota có hạn và executor phải kiểm soát được.

## 3. Thiết kế đang áp dụng

```text
Vietnamese question
  -> QuerySpec(entity, year, scope, target_unit)
  -> candidate table/row coordinates
  -> schema-only structured generation
  -> typed IR parse + grounding/dimension validation
  -> deterministic Pandas compile
  -> AST validation + isolated execution on evidence CSV
  -> declared answer == re-executed answer
```

| Insight | Hiện thực | Trạng thái |
|---|---|---|
| schema-only grounding | `generation/prompt.py` | `applied`, test cấm lộ source number |
| coordinate/cell operand | `programs/ir.py` | `applied` cho scalar binary IR |
| aggregate/count/rank | `AggregateExpr`, `CountIfExpr`, `ArgExtremumExpr` | `applied` baseline |
| dimension checking | `programs/dimensions.py` | `applied` cho typed expression |
| deterministic compilation | `programs/compiler.py` | `applied` |
| AST allowlist + process timeout | `programs/executor.py` | `applied`; POSIX memory optional, network/FS còn thiếu |
| re-execution | `submission/validate.py` | `applied` |
| checkpoint/error isolation | `scripts/50_generate_programs.py` | `applied` |
| LLM structured output → typed IR | `programs/serde.py`, generation runner | `applied` code/test; GPU smoke pending |
| panel coverage | dynamic candidate budget + ticker/year selected-cell gate | `applied` baseline; semantic hard-panel GPU pending |

## 4. IR mục tiêu và thứ tự triển khai

| Nhóm | Operator | Gate |
|---|---|---|
| scalar | `cell`, literal, `+ - * /` | đã có; division-by-zero phải fail |
| aggregate | `sum`, `mean`, `min`, `max`, `count_if` | unit đồng nhất và coverage operands |
| temporal | `diff`, `growth_pct`, CAGR | đúng start/end year, denominator convention |
| comparison | ratio/share, argmin/argmax | tie policy và entity balance |
| panel | nhiều ticker × nhiều năm | coverage matrix; không để một ticker chiếm hết evidence |

LLM chỉ sinh IR/coordinate; compiler tạo Pandas sau grounding/unit/dimension gate. Đây vẫn chưa phải
bằng chứng hard-tier cho tới khi panel coverage và Qwen GPU smoke/full run có log thật.

## 5. Vòng kiểm chứng bắt buộc

1. Validate JSON schema và tên biến.
2. Compile/parse AST, từ chối statement, import, attribute/call ngoài allowlist.
3. Thực thi trên đúng CSV đã package.
4. Answer phải hữu hạn và khớp re-execution theo tolerance cấu hình.
5. Kiểm provenance: mỗi bảng thuộc `relevant_docs`, mỗi biến có CSV, không path traversal.
6. Kiểm unit/dimension: cộng/trừ cùng dimension; ratio/percent không nhân/chia 100 hai lần.
7. Với câu đa thực thể, kiểm đủ ticker/year trước khi nhận candidate.

## 6. Ablation có thứ tự

1. raw Pandas expression vs typed scalar IR;
2. schema-only vs schema + row labels;
3. one-shot vs execution-guided retry một lần;
4. one candidate vs execution-consistency 3 candidates;
5. AWQ 4-bit vs BF16/FP16 nếu runtime cho phép.

Không chạy test-time scaling trước khi single-candidate pipeline vượt schema/execution gates; nếu
không chỉ tạo thêm nhiều biến thể của cùng một lỗi retrieval.

## 7. Các claim bị loại khỏi bản cũ

- Không dùng một taxonomy “5 tier chính thức”: release không có difficulty label; companion chỉ mô tả
  bốn tier và đó không phải gold field.
- Không khẳng định few-shot theo tier sẽ tăng điểm nếu chưa có ablation ViFinQA.
- Không dùng các paper tương lai/chưa xác minh metadata làm căn cứ cho self-correction.
- Không gọi executor AST hiện tại là OS sandbox; timeout/memory/network isolation còn thiếu.
