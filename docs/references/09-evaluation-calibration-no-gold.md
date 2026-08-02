# Evaluation, calibration và no-gold protocol

## 1. Sự thật hiện tại

Public ViFinQA release không có gold answer, qrels hay dev split. Parser invariants, retrieval route
coverage và execution success là QA hệ thống, không phải accuracy. Không được promote dense/reranker,
prompt hoặc repair policy chỉ vì proxy tăng.

## 2. Xây qrels bằng pooling

Theo thực hành pooling của [NIST TREC](https://trec.nist.gov/data/reljudge_eng.html), tạo pool từ các
run đa dạng để giảm thiên lệch về một retriever:

1. khóa question list, corpus/manifest hash và split trước khi gán nhãn;
2. lấy union top-k từ BM25 original, BM25 accentless, dense, late-interaction và reranker;
3. thêm route candidates có entity/year/scope exact nhưng chưa xuất hiện trong top-k;
4. deduplicate theo `table_ref` nhưng giữ run/rank/source score;
5. hai annotator độc lập gán `not relevant`, `partial`, `sufficient`, cùng cell coordinates;
6. disagreement được adjudicate; lưu rationale và version qrels;
7. bổ sung pool từ run mới rồi chỉ đánh giá lại khi unjudged@k vượt ngưỡng đã định.

Template 100 câu hiện chỉ là sampling frame. Trước khi có nhãn kép và adjudication, nó không phải dev
set. Cần báo inter-annotator agreement nhưng không dùng agreement cao thay accuracy.

## 3. Gold schema nhiều tầng

Mỗi case tối thiểu cần:

- question ID và normalized QuerySpec;
- answerability state + lý do;
- required entity/year/scope;
- relevant report/table/cell coordinates;
- source unit/scale/sign;
- ordered intermediate facts/cohort membership;
- typed operator/program và formula definition;
- exact high-precision result + output rounding/tolerance.

Ba tầng metric phải tách riêng:

| Tầng | Metric chính | Chẩn đoán |
|---|---|---|
| retrieval | sufficient-evidence Recall/F2@k, route coverage | MRR/nDCG, unjudged@k, entity/year/scope errors |
| reasoning | program exact/semantic execution, step alignment | cell/set/formula/unit accuracy |
| end-to-end | answer exact/tolerance accuracy | coverage, latency, failure class |

`gold-table program` là upper-bound reasoning; `retrieved-table deterministic program` tách retrieval;
`retrieved-table generated program` là end-to-end. Không so hai run khác cả retrieval lẫn generator rồi
gán nguyên nhân cho một component.

## 4. Split chống leakage

Random question split không đủ vì filings và generator template lặp cấu trúc. Dùng grouped split theo:

- company/ticker;
- report year và adjacent-year window;
- normalized question/template fingerprint;
- table/header fingerprint;
- metric/formula family.

Tạo ít nhất ba slice: interpolation (entity/year đã thấy), entity holdout và temporal holdout. Synthetic
data chỉ được sinh từ train partitions sau khi split. Deduplicate gần bằng question/table fingerprints;
không tune bằng public/private leaderboard probing. Các cảnh báo contamination được củng cố bởi
[NLP evaluation leakage review](https://arxiv.org/abs/2310.18018) và
[Benchmarking Benchmark Leakage](https://arxiv.org/abs/2404.18824).

## 5. Unanswerable và selective prediction

Answerability là state machine:

```text
ANSWERABLE
MISSING_IN_CORPUS
RETRIEVAL_INCOMPLETE
PARSE_OR_UNIT_AMBIGUOUS
CONFLICTING_EVIDENCE
EXECUTION_OR_MODEL_FAILURE
```

[Selective QA under domain shift](https://aclanthology.org/2020.acl-main.503/) và
[The Art of Abstention](https://aclanthology.org/2021.acl-long.84/) cho thấy confidence/abstention cần
được đánh giá như một quyết định riêng. Báo risk–coverage curve, selective accuracy tại các coverage
level đã khóa và calibration error/Brier score cho answerability. Raw LLM logprob không đủ; calibrator
dùng feature có thể audit: route completeness, retrieval margin, constraint-card match, grounding,
unit/formula gates và cross-path agreement.

Counterfactual unanswerable được sinh bằng perturb entity/year/metric hoặc xóa một required fact, như
hướng của [GBFR](https://aclanthology.org/2026.acl-long.1273/). Phải kiểm rằng perturbation thực sự
không còn answer trong corpus, tránh false-unanswerable.

Conformal methods như [CONFLARE](https://arxiv.org/abs/2404.04287) chỉ là `ablation`: coverage guarantee
phụ thuộc exchangeability; entity/year shift và leaderboard distribution có thể vi phạm giả định.

## 6. LLM-as-judge và reference-free metrics

[ARES](https://aclanthology.org/2024.naacl-long.20/) và
[RAGChecker](https://arxiv.org/abs/2408.08067) hữu ích để chẩn đoán context relevance/faithfulness,
nhưng không thay cell-level qrels, executable program và numeric gold. Nếu rule cấm proprietary model,
judge cũng phải tuân thủ. Judge prompt/model/revision phải được pin và audit trên human labels trước.

## 7. Thống kê và promotion gate

- report macro và micro; macro theo question để multi-table case không lấn át;
- bootstrap CI theo question; paired bootstrap/permutation cho run cùng split;
- report effect size và số case, không chỉ p-value;
- slice theo entity count, year count, scope, operator family, OCR severity và answerability;
- không tune threshold trên test/private;
- promote khi metric chính tăng, CI/risk chấp nhận được, không giảm critical slice, và latency/VRAM
  trong budget.

Với manual set nhỏ, CI có thể rộng; kết luận đúng là “chưa đủ bằng chứng”, không phải chọn run có điểm
cao hơn vài case.
