# Deep research matrix

**Nguyên tắc nguồn:** ưu tiên đặc tả, dataset card/repo chính thức, paper/arXiv/ACL/ACM và model
card chính chủ. Blog thứ cấp chỉ dùng để tìm nguồn, không dùng làm bằng chứng kỹ thuật cuối.

## 1. Nguồn trực tiếp về ViFinQA

| Nguồn | Đã xác minh | Thông tin dùng | Cảnh báo |
|---|---|---|---|
| [ViFinQA HF](https://huggingface.co/datasets/AIGuruTinix/ViFinQA) | có + tải local | 1.012 câu, 1.973 report, 100 ticker, 2015–2025, public fields | card ghi 143.815 normalized tables và không có gold |
| [Companion repo](https://github.com/DSKT-NOWJ/ViFinQA) | commit `9a046de` | canonical `doc|table_N`, retrieval/eval configs, four tiers | README ghi 146.246 nhưng `docs/data.md` còn 143.815 |
| đặc tả local | có | metric, submission schema, model/submission rule | `doc|350` và tolerance chưa giải thích |
| [TiniX OCR source](https://huggingface.co/datasets/tinixai/ocr_annual_financials) | card | nguồn corpus rộng hơn | không thay cho subset ViFinQA đã audit |

Quyết định: số liệu release phải lấy từ audit local + hash; upstream text chỉ là cross-check.

## 2. Financial numerical QA và executable reasoning

| Công trình | Kết luận liên quan | Áp dụng |
|---|---|---|
| [FinQA](https://arxiv.org/abs/2109.00122), Chen et al. 2021 | financial QA cần gold reasoning program, multi-step numerical reasoning còn xa expert | typed program, execution trace |
| [TAT-QA](https://arxiv.org/abs/2105.07624), Zhu et al. 2021 | hybrid table/text; operators add/subtract/multiply/divide/count/compare/sort | operator inventory, hybrid evidence |
| [DataFrame QA](https://arxiv.org/abs/2401.15463), Ye et al. 2024 | LLM sinh Pandas trên schema, execution-based evaluation | schema-only prompt + execution; không lấy benchmark score làm dự báo domain |
| [RePanda](https://arxiv.org/abs/2503.11921), Chegini et al. 2025 | executable Pandas tăng khả năng kiểm chứng và có thể distill vào 7B | 7B coder + executable IR |
| [Text-to-Pipeline](https://arxiv.org/abs/2505.15874), Ge et al. 2025 | iterative predict/execute với feedback tốt hơn one-shot cho pipeline nhiều bước | staged IR/execution, controlled repair |
| [DS-1000](https://arxiv.org/abs/2211.11501) | realistic data-science code evaluation cần execution + perturbation | metamorphic/differential tests |
| [API-assisted Table QA](https://arxiv.org/abs/2310.14687) | multi-index Pandas + unified API cho varied table structures | giữ coordinate/schema, cân nhắc API IR |
| [LOFin/HiREC](https://aclanthology.org/2025.findings-acl.855/) | standardized filings gây near-duplicate; hierarchical retrieval + evidence curation | route document trước table; finance constraints |
| [MuDABench](https://aclanthology.org/2026.findings-acl.341/) | multi-document analytical QA; đo intermediate-fact coverage | coverage của operand/cohort trước final answer |
| [FinChain](https://aclanthology.org/2026.acl-long.662/) | parameterized symbolic templates + executable traces + step alignment | program-first test/synthetic data |
| [GBFR](https://aclanthology.org/2026.acl-long.1273/) | metric graph, cross-path verification, safe abstention | metric dependency graph nhỏ; counterfactual unanswerable |
| [SEC-FinTables](https://aclanthology.org/2026.findings-acl.764/) | totals/components logical inconsistency, cell-level diagnostics | consistency signal, không tự sửa raw OCR |

Điểm rút ra: program chạy được chưa đủ; phải đúng cell, unit và logic. Cần đánh giá gold-table program
riêng để tách lỗi retrieval khỏi reasoning.

### 2.1 Coverage local đã kiểm

Companion có 9 scenario trung gian và 70 typed hard grammar; release không có mapping câu→template.
Terminal hard gồm 32 lookup, 10 maximum, 9 share, 6 count, 6 difference, 5 mean, 2 ratio-of-sums,
nhưng các chuỗi trước terminal còn filter/rank/set/temporal/formula. Vì vậy plan dùng họ grammar làm
coverage contract, không dùng terminal distribution làm difficulty distribution. Xem
[taxonomy chi tiết](07-bai-toan-taxonomy.md).

## 3. Table retrieval

| Công trình | Kỹ thuật | Quyết định/ablation |
|---|---|---|
| [Dense table retrieval](https://arxiv.org/abs/2103.12011) | dual encoder retrieval trên table corpus | dense baseline |
| [THYME](https://arxiv.org/abs/2503.02251), Li et al. 2025 | field-aware hybrid matching; title/header/cell có matching preference khác nhau | multi-view index |
| [QGpT](https://arxiv.org/abs/2508.06168), Liang et al. 2025 | synthetic questions từ partial table để align table/query | semantic-question view, chỉ sau leakage check |
| [DCTR](https://arxiv.org/abs/2603.07146), Kosiuk et al. 2026 | typed query decomposition + connectivity awareness | typed multi-table route; sửa mô tả cũ “hai dense index” |
| [Adaptive Table Retrieval](https://arxiv.org/abs/2605.18766), Kim et al. 2026 | adaptive threshold + sliding-window rerank thay fixed-k | calibrated stopping ablation |
| [T-RAG](https://aclanthology.org/2026.findings-acl.1902/), Zou et al. 2026 | hierarchical memory, multi-stage retrieval, graph-aware context | hierarchy/entity-year graph, không copy nguyên framework |
| [T2-RAGBench](https://aclanthology.org/2026.eacl-long.8/), Strich et al. 2026 | hybrid BM25/dense hiệu quả trên text+table benchmark | củng cố BM25+dense RRF, vẫn phải đo ViFinQA |
| [Efficient visual table retrieval](https://aclanthology.org/2026.findings-eacl.226/), Xu et al. 2026 | visual-text retrieve → MLLM rerank/reason | chỉ liên quan nếu corpus chuyển sang table image |
| [FinCARDS](https://aclanthology.org/2026.findings-acl.1244/) | card entity/metric/period/value + constraint reranking | finance-aware auditable rerank |
| [CRAFT](https://aclanthology.org/2026.acl-long.149/) | sparse shortlist rồi dense rerank, training-free | cascade hợp compute; không dùng proprietary summary generator |
| [PieTa](https://aclanthology.org/2026.acl-long.1460/) | iterative window/multi-resolution subtable selection | chọn giao row×column trên gold table trước |

Không paper nào thay thế qrels ViFinQA. Kết quả ngoài domain chỉ dùng để chọn ablation đáng thử.

## 4. Vietnamese retrieval/embedding

| Nguồn | Bằng chứng | Cách dùng |
|---|---|---|
| [VN-MTEB](https://arxiv.org/abs/2507.21500), Pham et al. 2025 | 41 dataset, 6 task Vietnamese embedding | shortlist/evaluate embedding |
| [Vietnamese IR benchmark](https://arxiv.org/abs/2503.07470), Nguyen et al. 2025 | benchmark retrieval/reranking Việt + InfoNCE variant | error slice Vietnamese IR |
| [Vietnamese multi-domain IR](https://aclanthology.org/2026.findings-eacl.110/), Nguyen & Quan 2026 | so lexical/neural-sparse/late-interaction/dense/hybrid trên 6 domain | không mặc định model lớn/hybrid thắng; giữ BM25 control |
| [BGE-M3 card](https://huggingface.co/BAAI/bge-m3) | >100 languages, dense/sparse/multi-vector, 8192 token, MIT | dense default |
| [bge-reranker-v2-m3 card](https://huggingface.co/BAAI/bge-reranker-v2-m3) | multilingual cross-encoder, Apache-2.0 | reranker default |

VN-MTEB cho thấy BGE-M3 là candidate đáng thử, nhưng đây không phải financial-table retrieval score.
BM25 accentless vẫn cần vì OCR/diacritics; model cuối phải được chọn bằng ablation ViFinQA.

## 5. Model/program generation

| Nguồn | Fact đã kiểm | Hệ quả |
|---|---|---|
| [Qwen2.5-Coder-7B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-AWQ) | base 7,61B; AWQ 4-bit; Apache-2.0; code model | hợp rule strict |
| [Qwen2.5-7B-AWQ](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ) | 7,61B; AWQ; Vietnamese trong multilingual list | NLU fallback |
| [vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/) | JSON schema/structured output hiện hành | `response_format=json_schema` |
| [vLLM GPU install](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/) | Linux/CUDA constraints | không cài native Windows |
| [vLLM PyPI](https://pypi.org/project/vllm/) | 0.25.1 kiểm tại 02/08/2026; Python 3.10–3.14 | pin notebook rồi lưu environment |

Không dùng Qwen2.5-Coder-14B vì card thực tế 14,7B. Không dùng proprietary model cho bất kỳ stage
nào nếu rule cấm closed model.

## 6. Runtime hosted

- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html): resource không guaranteed,
  usage/GPU type thay đổi, VM có lifetime; free runtime thường tối đa 12 giờ tùy khả dụng.
- [Kaggle notebook docs](https://www.kaggle.com/docs/notebooks): dùng làm nguồn policy hiện hành;
  UI/quota cần kiểm lúc chạy vì page không crawl ổn định.

Quyết định: detect GPU/RAM/disk; smoke; checkpoint; không hứa exact hardware/quota.

## 7. Văn bản kế toán Việt Nam

| Văn bản | URL chính thức | Vai trò |
|---|---|---|
| TT200/2014/TT-BTC | [lịch sử VBPL](https://vbpl.vn/TW/Pages/vbpq-lichsu.aspx?ItemID=66801) | schema doanh nghiệp lịch sử; hết hiệu lực 01/01/2026 |
| TT99/2025/TT-BTC | tra cứu VBPL | thay TT200 từ 2026; ngoài phần lớn corpus nhưng cần ghi trạng thái |
| TT49/2014/TT-NHNN | [VBPL NHNN](https://vbpl.vn/nganhangnhanuoc/Pages/vbpq-thuoctinh.aspx?ItemID=52646) | BCTC tổ chức tín dụng |
| TT334/2016/TT-BTC | [VBPL](https://vbpl.vn/TW/Pages/vbpq-thuoctinh.aspx?ItemID=118869) | mẫu BCTC/doanh nghiệp theo quy định liên quan |

Văn bản pháp lý chỉ giúp canonical label/scope/unit; con số và evidence vẫn lấy từ corpus.

## 8. Research-to-code mapping

| Insight | Module/config |
|---|---|
| field-aware/multi-view | `indexing/manifest.py`, configs retrieval |
| hybrid BM25+dense | `retrieval/bm25.py`, `dense.py`, `fusion.py` |
| cross-encoder | `retrieval/rerank.py` |
| entity balance | `balanced_round_robin` |
| executable Pandas | `programs/`, `evidence/store.py` |
| schema-only grounding | `generation/prompt.py` |
| execution validation | `submission/validate.py` |
| reproducibility | scripts 00/10/20/30/40/41 + runbook |
| finance constraint cards | QuerySpec + manifest metadata; `TODO` rerank feature |
| metric dependency/formulas | typed IR + versioned formula registry; `TODO` mở rộng |
| intermediate-fact evaluation | qrels schema + program traces; `TODO` human labels |
| safe abstention | structured failure state + risk–coverage; `TODO` calibration |
| adversarial sandbox | current executor chưa đạt; xem file 12 |

## 9. Không áp dụng hoặc hoãn

- Vision OCR/MLLM: hoãn vì release đã có text + inline HTML.
- Fine-tune retriever bằng synthetic question: hoãn tới khi có leakage-safe split/qrels.
- Query rewriting: hoãn vì có thể drift financial term; chỉ ablation.
- Adaptive k production: hoãn tới khi score calibration có dev labels.
- LLM self-correction nhiều vòng: hoãn; tăng nondeterminism/compute và khó audit.
- 14B/closed frontier model: không hợp compliance hiện tại.
- RAPTOR/GraphRAG global summarization: hoãn vì exact scalar/evidence retrieval là critical path.
- Raw arbitrary Python production: loại; giữ typed DSL. Raw code chỉ là research ablation cô lập.
