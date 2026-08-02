# Báo cáo kiểm thử chất lượng

**Ngày:** 02/08/2026

**Môi trường:** Windows, Python 3.12.10, CPU inference.

Máy có RTX 3050 4 GiB ở chế độ WDDM nhưng environment hiện dùng `torch 2.13.0+cpu`,
`torch.cuda.is_available()=false`; không dùng cấu hình này làm bằng chứng vLLM/CUDA.

## 1. Kết quả hiện tại

| Gate | Kết quả |
|---|---|
| pytest | 56 passed |
| Ruff | all checks passed |
| Ruff format | 71 files formatted |
| mypy strict | success, 53 source files (`src` + `scripts`) |
| pip check | no broken requirements |
| compileall | `src` + `scripts` pass |
| full corpus markup | 146.246 opening = closing = matched |
| full manifest unique refs | 146.246/146.246 |
| parsed empty/malformed | 0/0 |
| context leak từ bảng trước | regression pass; previous `</table>` là boundary |
| mixed-unit ACV/USD | table `UNKNOWN`, cell `USD`, 6.15569834 triệu USD |
| QuerySpec target unit unknown | 0/1.012 |
| entity unresolved | 1 câu global không nêu ticker |
| multi-entity / multi-year | 289 / 360 câu |
| BM25 index count | 146.246 |
| metadata-routed empty retrieval | 0/1.012 |
| available ticker×year routes missing | 0 |
| retrieval refs outside/wrong metadata route | 0/0 |
| next-report comparative fallback | 3: PDR-2024, BSR-2016, BID-2021 |
| retrieval QC failures | 0 mọi nhóm; report `passed=true` |
| full retrieval wall time / SHA-256 | 54,9 s / `99bcd364...38bf09` |
| real-data q1 execution | 208253.201298 triệu VND |
| qrels template v2 | 100 unique/77 strata, 100% unlabeled, schema valid |
| qrels pool control | 100 questions/1.998 candidates, schema valid, no duplicate ref/question |
| notebook JSON validation | 2 notebook hợp lệ nbformat 4 |
| notebook code compile | 15/15 code cells pass |
| bibliography | 47/47 unique keys, balanced braces |
| local Markdown links | 0 missing trong README/docs/annotations |

## 2. Test coverage theo tầng

- number parsing: locale, negative parentheses, dash, OCR refusal;
- table segmentation: ID/page/context boundary + numbered-section carry;
- table parsing: rowspan/colspan/header/unit;
- NLU: longest unit phrase, explicit scope, prior-period/range year, alias shadowing/proper-name gate,
  target/counterparty, multi-parent comparison và cohort preservation;
- metrics: F2/tolerance;
- manifest: provenance/hash and no source numbers in retrieval view;
- BM25: accent tolerance + exact metadata-prefilter even với candidate pool nhỏ;
- fusion: RRF deterministic + entity×year round-robin;
- program: scalar/aggregate/count-if/arg-extremum compiler, dimension checker, malicious syntax;
- structured program: recursive JSON Schema/parser, extra-field rejection, grounding/unit/target scale;
- isolation: child-process success, timeout termination; submission re-execution qua isolated worker;
- submission: schema, re-execution, ZIP root;
- generation prompt: coordinate visible, source number hidden;
- corpus integration: real VJC/VND và mixed-table ACV/USD answer conversion.
- qrels: v2 schema/fingerprint validation; pooling union/depth/provenance/determinism/no-duplicate.

## 3. Chưa được kiểm vì phụ thuộc ngoài

- dense BGE-M3 full index;
- reranker full run;
- Qwen/vLLM generation;
- official retrieval/answer/execution accuracy vì release không có gold;
- dashboard table-ref/tolerance contract;
- final 1.012-prediction ZIP.

Local preflight RUN-012 đã chạy thêm BGE-M3 offline CPU smoke trên 100 bảng và positive-route q471;
đây là compatibility evidence, không phải dense quality/latency benchmark. Full dense vẫn cần CUDA.

Các mục này không được ghi “passed” cho tới khi artefact/log thật tồn tại.

## 4. Regression gates cần bổ sung

1. tables with unit anchor after table và propagation cho block unit nằm trong body;
2. EUR/ngoại tệ khác và currency conversion chỉ khi có tỷ giá evidence;
3. report-year vs comparative-column-year typed semantics;
4. population intent cho nhóm ngành/toàn bộ công ty không liệt kê ticker;
5. semantic hard-panel coverage và GPU structured-output compatibility;
6. read-only filesystem/network/CPU isolation; POSIX memory-limit integration smoke;
7. manifest/dense embedding shard-resume;
8. dense count/dimension/normalization full GPU;
9. model revision/cutoff registry thực tế;
10. full ZIP clean-room validation.
11. diverse qrels pool với BM25/dense/late/rerank, double annotation và adjudication.

## 5. Tiêu chuẩn báo lỗi

Mọi failure full run phải có: question ID, stage, error type, message, candidate refs, config hash và
model revision. Không bỏ qua exception để điền answer 0 hoặc empty table. Submission validator phải
fail closed.
