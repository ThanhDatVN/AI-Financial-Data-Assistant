# Kỹ thuật mới và ma trận ablation

Tài liệu này chuyển research thành quyết định có thể kiểm thử. Claim của paper chỉ mô tả setting của
paper; không được dùng như kết quả dự kiến trên ViFinQA.

## 1. Financial numerical QA

| Nguồn sơ cấp | Bài toán/kỹ thuật | Bài học cho ViFinQA | Trạng thái |
|---|---|---|---|
| [MultiHiertt](https://aclanthology.org/2022.acl-long.454/) | nhiều bảng phân cấp + text, supporting facts, symbolic reasoning | lưu hierarchy và gold intermediate facts | `ablation` |
| [ConvFinQA](https://aclanthology.org/2022.emnlp-main.421/) | numerical reasoning có lịch sử hội thoại | hữu ích nếu dashboard có follow-up; release hiện không có | `deferred` |
| [DocFinQA](https://aclanthology.org/2024.acl-short.42/) | QA trên full filing rất dài | route document trước table/cell | `ablation` |
| [LOFin/HiREC](https://aclanthology.org/2025.findings-acl.855/) | near-duplicate standardized filings, doc→passage retrieval và evidence curation | rất sát lỗi cùng mẫu giữa company/year | `next` |
| [MuDABench](https://aclanthology.org/2026.findings-acl.341/) | multi-document extraction + aggregate, intermediate-fact coverage | đo evidence coverage trước final answer | `next` |
| [FinChain](https://aclanthology.org/2026.acl-long.662/) | symbolic template + executable program + step alignment | đánh giá trace từng bước, sinh case không nhiễm corpus | `next` |
| [GBFR](https://aclanthology.org/2026.acl-long.1273/) | metric graph, cross-path verification, safe abstention | metric dependency graph nhỏ + distinguish missing/retrieval failure | `ablation` |

[FinanceBench](https://arxiv.org/abs/2311.11944),
[FinTextQA](https://aclanthology.org/2024.acl-long.328/) và
[FinLFQA](https://aclanthology.org/2025.findings-emnlp.908/) là benchmark attribution/long-form hữu
ích cho evidence provenance, nhưng ViFinQA yêu cầu scalar executable answer nên không dùng LLM judge
thay execution.

## 2. Chọn bảng và subtable

| Nhóm | Candidate | Khi có lợi | Rủi ro/chi phí |
|---|---|---|---|
| lexical | BM25 original + accentless | exact metric/entity/year, OCR nhẹ | đồng nghĩa/near-duplicate |
| learned sparse | [SPLADE v2](https://arxiv.org/abs/2109.10086) | mở rộng term nhưng vẫn inverted index | model/index mới; cần qrels |
| dense | BGE-M3 | paraphrase tiếng Việt/financial labels | semantic false positive |
| late interaction | [ColBERTv2](https://arxiv.org/abs/2112.01488), multilingual [Jina-ColBERT](https://aclanthology.org/2024.mrl-1.11/) | matching token/cell tinh hơn vector đơn | storage/latency |
| CPU cascade | [SPLATE](https://arxiv.org/abs/2404.13950) | sparse shortlist + late interaction | implementation/profile |
| fusion | [RRF](https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/) | rank scale không đồng nhất | cần tune depth, không tune weight mù |
| finance rerank | [FinCARDS](https://aclanthology.org/2026.findings-acl.1244/) | constraint entity/metric/period/value | parser card sai sẽ hard-filter nhầm |
| cascaded table | [CRAFT](https://aclanthology.org/2026.acl-long.149/) | sparse shortlist trước dense | paper dùng generated title/summary; không dùng Gemini do rule |
| subtable | [PieTa](https://aclanthology.org/2026.acl-long.1460/) | bảng lớn, evidence nằm giao row×column | iterative cost; cần preserve coordinate |

[Nghiên cứu IR tiếng Việt đa miền 2026](https://aclanthology.org/2026.findings-eacl.110/) đánh giá
lexical, neural-sparse, late-interaction, dense và hybrid; kết luận thực dụng là phải benchmark theo
domain, model lớn hơn hoặc hybrid không tự động thắng. Vì vậy BM25 vẫn là control bắt buộc.

Thứ tự ablation retrieval, mỗi lần đổi một biến:

1. BM25 metadata-routed;
2. BM25 multi-view;
3. dense;
4. BM25+dense RRF;
5. RRF + cross-encoder;
6. RRF + finance constraint cards;
7. SPLADE hoặc late interaction thay dense;
8. subtable selection trên gold table rồi mới end-to-end;
9. calibrated adaptive stopping sau khi có dev score.

## 3. Table reasoning và program synthesis

| Kỹ thuật | Nguồn | Quyết định |
|---|---|---|
| weak cell selection + aggregation | [TAPAS](https://aclanthology.org/2020.acl-main.398/) | benchmark idea; không phù hợp scale corpus nếu fine-tune từ đầu |
| executable SQL pretraining | [TAPEX](https://arxiv.org/abs/2107.07653), [OmniTab](https://aclanthology.org/2022.naacl-main.68/) | synthetic execution supervision sau leakage split |
| precomputed arithmetic cube | [TaCube](https://aclanthology.org/2022.emnlp-main.145/) | thử cho sum/difference/ratio phổ biến, tránh nổ tổ hợp |
| question-specific row filters | [ToolWriter](https://aclanthology.org/2023.emnlp-main.1003/) | typed filter op cho bảng dài |
| bind LM calls trong program | [BINDER](https://openreview.net/forum?id=lH1PV42cbF) | chỉ semantic label resolver; arithmetic luôn deterministic |
| decomposition | [DATER](https://arxiv.org/abs/2301.13808) | tách evidence, filter, formula; log intermediate state |
| iterative table operations | [Chain-of-Table](https://openreview.net/forum?id=4L0xnS4GQM) | ablation IR transformations; không mutate evidence gốc |
| program-of-thought | [PoT](https://arxiv.org/abs/2211.12588) | dùng compiler/executor; execution success chưa đủ |
| decompose/sanitize/code | [TabDSR](https://aclanthology.org/2025.findings-emnlp.169/) | thêm sanitize stage có audit trail |

Production path vẫn là `typed JSON DSL → deterministic compiler → executor`. Raw Python/SQL chỉ là
ablation vì khó kiểm quyền truy cập và dễ “chạy được nhưng sai logic”.

## 4. Constrained decoding và verification

- [PICARD](https://aclanthology.org/2021.emnlp-main.779/) chứng minh incremental constrained
  decoding có thể loại token không hợp grammar. Áp dụng bằng JSON Schema/CFG, không chỉ parse hậu kỳ.
- [XGrammar](https://arxiv.org/abs/2411.15100) và
  [Outlines](https://arxiv.org/abs/2307.09702) là candidate runtime cho CFG/regex; benchmark latency
  trên exact model/runtime trước khi chọn.
- Execution-guided refinement ([ACL 2024](https://aclanthology.org/2024.findings-acl.120/),
  [ExeSQL](https://aclanthology.org/2025.findings-emnlp.1320/)) chỉ repair syntax/schema/runtime.
  Không dùng execution success làm verifier ngữ nghĩa.
- [ProgCo](https://aclanthology.org/2025.acl-short.73/) cho thấy program-driven verification là một
  hướng, nhưng self-correction có thể sửa đáp án đúng thành sai. Chỉ cho một repair và đo paired.

Verifier đề xuất gồm bảy gate độc lập: schema, grounding, route coverage, dimension/unit, formula,
intermediate-set/cell coverage và final numeric/finite. Nếu có hai đường công thức độc lập, so sánh
cross-path; mismatch phải abstain hoặc chuyển manual review.

## 5. Long context, graph và multimodal

- [Lost in the Middle](https://arxiv.org/abs/2307.03172) cảnh báo model dài context vẫn nhạy vị trí;
  không nhồi toàn report vào prompt.
- [RAPTOR](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8a2acd174940dbca361a6398a4f9df91-Abstract-Conference.html)
  và [GraphRAG](https://www.microsoft.com/en-us/research/publication/from-local-to-global-a-graph-rag-approach-to-query-focused-summarization/)
  phù hợp global sensemaking/summarization. ViFinQA chủ yếu cần exact scalar và precise evidence nên
  giữ `deferred`; metric dependency graph không đồng nghĩa GraphRAG.
- [FinMRAGBench](https://aclanthology.org/2026.findings-acl.187/) và
  [FinRAGBench-V](https://aclanthology.org/2025.emnlp-main.211/) cho thấy cross-page/multimodal là
  ngách quan trọng nếu sau này nguồn chuyển về PDF/image. Corpus HTML hiện tại chưa biện minh chi phí.

## 6. Ngân sách mô hình ≤14B

- Generator mặc định 7B AWQ; không dùng Qwen2.5-Coder-14B vì model card ghi 14,7B.
- [AWQ](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html),
  [GPTQ](https://arxiv.org/abs/2210.17323) là inference ablation; chọn theo accuracy/latency/VRAM thực.
- [QLoRA](https://arxiv.org/abs/2305.14314) chỉ dùng sau khi synthetic/gold data vượt gate chất lượng.
- Speculative decoding có thể tăng tốc nhưng cần draft model và đo exact-output/VRAM; không ở critical
  path với chỉ 1.012 câu.

## 7. Thứ tự ưu tiên

| Ưu tiên | Hạng mục | Lý do |
|---|---|---|
| P0 | qrels + intermediate facts + leakage-safe split | không có thước đo thì mọi model change đều mù |
| P1 | cohort/temporal/dependent lookup IR | trực tiếp phủ grammar hard local |
| P1 | finance cards + hierarchical route | đánh đúng lỗi near-duplicate corpus |
| P1 | abstention/risk–coverage | tránh forced numeric answer khi thiếu data |
| P2 | SPLADE/late interaction/PieTa | promising nhưng cần ablation |
| P2 | synthetic supervision | chỉ sau split và executable validation |
| P3 | OCR vision, RAPTOR, GraphRAG, multi-agent | ngoài critical path hiện tại |
