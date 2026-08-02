# Khung working-notes paper có truy vết

Không điền số ước lượng hoặc claim từ paper ngoài miền vào cột kết quả. Mỗi số trong bản nộp phải có
`RUN_ID`, config/hash và log tái lập.

## 1. Title và abstract

Title gợi ý: **Evidence-grounded Table Retrieval and Typed Text-to-Pandas for Vietnamese Financial
Reports**.

Abstract chỉ hoàn thiện sau final run, gồm: task; corpus audit; phương pháp; kết quả ba metric; một
ablation chính; giới hạn. Không dùng “state-of-the-art” nếu không có baseline/cùng split.

## 2. Task và quy định

- input/output contract, ZIP/evidence/Pandas;
- Precision/Recall/F2, Answer Accuracy, Execution Accuracy;
- open/public model, ≤14B, cutoff 01/06/2026;
- các điểm dashboard còn phải xác nhận: table-ref grammar, tolerance, quota.

## 3. Dataset và audit

| Thuộc tính | Giá trị có log | Nguồn |
|---|---:|---|
| questions/reports/tickers | 1.012 / 1.973 / 100 | `RUN-001` |
| years | 2015–2025 | `RUN-001` |
| raw tables | 146.246 | `RUN-001/002` |
| table refs unique | 146.246 | `RUN-006` |
| empty/malformed | 0/0 | `RUN-006` |

Giải thích sai lệch 143.815 vs 146.246; ghi dataset revision/hash, CC BY-NC 4.0 và không có
train/dev/gold public.

## 4. Method

1. parser/provenance và long evidence CSV;
2. QuerySpec: entity role, year, scope, target unit;
3. exact metadata-prefilter, BM25, optional BGE-M3/reranker;
4. balance `ticker × report_year`, RRF;
5. schema-only generation không lộ source digits;
6. structured typed IR, grounding/dimension checker, compiler, isolated AST executor;
7. validator và clean-room packaging.

Sơ đồ phải chỉ rõ raw value chỉ đi từ parser → CSV → executor, không qua prompt LLM.

## 5. Experimental setup

| Mục | Phải ghi |
|---|---|
| code | Git SHA + worktree clean/diff artifact |
| data | repo/revision/checksum |
| model | base/repo thực dùng@SHA, params, license, quantization, release date |
| runtime | Python/torch/CUDA/vLLM/GPU/RAM |
| inference | seed, temperature, max tokens, context, batch/TP |
| retrieval | candidate depth, final k, fusion/rerank config |
| evaluation | qrels/split hash, tolerance source, macro formulas |

## 6. Kết quả — chỉ điền từ run log

| Run | Retrieval F2 | Answer Acc | Execution Acc | Latency | Compliance |
|---|---:|---:|---:|---:|---|
| BM25 | `TODO` | `TODO` | `TODO` | `RUN-007` | CPU/open |
| + dense/RRF | `TODO` | `TODO` | `TODO` | `TODO` | pin SHA |
| + reranker | `TODO` | `TODO` | `TODO` | `TODO` | pin SHA |
| final | `TODO` | `TODO` | `TODO` | `TODO` | checklist pass |

Không biến `0 empty route` thành Recall/F2. RUN-007 hiện chỉ chứng minh route health/invariants.

## 7. Ablation tối thiểu

- metadata prefilter on/off;
- ticker-only vs ticker×year balance;
- BM25 vs dense vs RRF;
- reranker on/off;
- raw expression vs typed IR;
- AWQ vs BF16/FP16 nếu có cùng hardware/setting;
- gold-table program vs end-to-end retrieved-table.

Báo paired delta trên cùng question set; thêm confidence interval/bootstrap nếu qrels đủ lớn.

## 8. Error analysis

Slice theo entity role, number of entities/years, scope, target unit, operator, OCR/unit unknown. Nhãn
lỗi: route, wrong table/cell, temporal, scope, unit, operator, syntax, execution, missing coverage,
contract/package.

## 9. Compliance và external resources

- model registry ở `references/06` phải hết `TODO` cho model thực dùng;
- kê khai mọi external data/model/tool, license và revision;
- tuyên bố không dùng proprietary LLM endpoint;
- TT200/TT49/TT334 chỉ hỗ trợ label/schema, không thay số corpus;
- đính kèm prompt/schema hash và environment lock.

## 10. Limitations và reproducibility

Nêu thẳng: public no gold; table-ref/tolerance dashboard unresolved; OCR text-only; typed population,
panel coverage và OS-level network/read-only isolation chưa hoàn thiện nếu tới final vẫn còn; hosted GPU
không guaranteed. Cung cấp lệnh
runbook, notebook, artefact manifest và SHA-256 của submission ZIP.

## 11. Checklist trước gửi paper

- [ ] mọi số có RUN_ID/log;
- [ ] citation xuất phát từ `bibliography.bib`, không còn nguồn unverified;
- [ ] model revision/license/cutoff đã kiểm;
- [ ] external resources đầy đủ;
- [ ] không claim official metric từ manual qrels;
- [ ] limitation khớp trạng thái code;
- [ ] submission ZIP/hash trong paper trùng dashboard log.
