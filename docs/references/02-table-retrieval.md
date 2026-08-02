# 02 — Truy hồi bảng: metadata, sparse, dense và hợp nhất

## 1. Bằng chứng nghiên cứu

| Nguồn gốc | Kỹ thuật | Hệ quả cho ViFinQA |
|---|---|---|
| [Dense Table Retrieval](https://arxiv.org/abs/2103.12011) | dual-encoder cho table corpus | dense là baseline bổ sung, không thay lexical |
| [THYME](https://arxiv.org/abs/2503.02251), Li et al. 2025 | field-aware sparse+dense theo title/header/cell | manifest giữ nhiều view thay vì nối mù |
| [QGpT](https://arxiv.org/abs/2508.06168), Liang et al. 2025 | sinh question từ partial table | chỉ ablation sau khi có leakage-safe evaluation |
| [DCTR](https://arxiv.org/abs/2603.07146), Kosiuk et al. 2026 | typed query decomposition và connectivity | route riêng cho multi-entity/multi-table |
| [Adaptive Table Retrieval](https://arxiv.org/abs/2605.18766), Kim et al. 2026 | adaptive threshold/sliding-window rerank | thử calibrated stopping khi có qrels |
| [T-RAG](https://aclanthology.org/2026.findings-acl.1902/), Zou et al. 2026 | hierarchy, multi-stage retrieval, graph-aware context | entity-year hierarchy và coverage graph |
| [T2-RAGBench](https://aclanthology.org/2026.eacl-long.8/), Strich et al. 2026 | benchmark buộc retrieve trước numerical QA; hybrid hiệu quả trong setting paper | củng cố BM25+dense ablation, không ngoại suy score |

Visual-table retrieval không nằm critical path vì release đã có text/HTML; chỉ xem lại nếu audit phát
hiện bảng quan trọng mất cấu trúc và có ảnh/PDF hợp lệ.

## 2. Phân rã retrieval đúng với corpus

```text
Question
  -> metadata route: ticker × report year × explicit scope
  -> BM25 over accent-preserving + accentless multi-view text
  -> optional BGE-M3 dense candidates
  -> per-entity balanced merge
  -> RRF / optional reranker
  -> evidence coverage + confidence policy
```

Metadata không phải soft feature khi câu hỏi nêu rõ ticker/year: route sai làm search toàn bộ 146.246
bảng vừa chậm vừa tăng false positive. Scope chỉ được lọc khi nêu rõ; lúc đó nhận scope đích và
`unknown`, không nhận scope đối nghịch. Không mặc định `consolidated`.

## 3. Multi-view index

| View | Nội dung | Tác dụng |
|---|---|---|
| identity | ticker, company aliases, year, scope, doc_id | route/diagnostic |
| context | section text trước bảng, page | tìm tiêu đề/thuyết minh |
| header | header/cột/mốc thời gian | temporal/schema linking |
| label | nhãn dòng gốc + bỏ dấu | lexical match chống OCR |
| semantic | text đã bỏ chữ số nguồn | dense retrieval không làm LLM chép số |

Giữ raw table/provenance riêng để xuất evidence; không index một bản “normalized table” không có
checksum/coordinate quay về nguồn.

## 4. Hợp nhất và câu đa thực thể

RRF hợp nhất thứ hạng sparse/dense mà không giả định score cùng thang. Với nhiều ticker, mỗi ticker
được route riêng rồi `balanced_round_robin`; nếu hợp nhất tất cả candidate trước, một ticker có nhiều
lexical hit sẽ lấn ticker khác. Test hiện có kiểm BAB/SGB và DTK/HND xen kẽ.

`top_k` submission không được cố định từ paper. F2 phạt bảng thừa, nhưng câu multi-table cần đủ recall;
k phải dựa trên coverage/confidence. Chưa có qrels thì giữ candidate nội bộ rộng, còn bảng nộp chỉ
được chốt sau execution/evidence selection.

## 5. Trạng thái hiện thực

| Thành phần | Module/artefact | Trạng thái |
|---|---|---|
| full manifest | `indexing/manifest.py` | 146.246 ref duy nhất, 0 malformed |
| BM25 | `retrieval/bm25.py` | 146.246 records, full CPU run |
| multi-entity balance | `retrieval/fusion.py` | applied + regression tests |
| BGE-M3/FAISS | `retrieval/dense.py` | code/notebook có; chưa full GPU run |
| cross-encoder | `retrieval/rerank.py` | code có; chưa benchmark |
| adaptive stopping | chưa có | deferred tới khi có calibration labels |
| synthetic question index | chưa có | deferred vì leakage/evaluation risk |

## 6. Evaluation đúng cách

Release không có qrels, nên `0 empty candidate` chỉ là health check, **không phải Recall**. Cần ba lớp:

1. gold-table fixtures thủ công, khóa trước khi tune;
2. slice theo single/multi entity, year, scope, unit và OCR noise;
3. dashboard score với submission log/hash khi contract đã xác nhận.

Các metric offline cần báo `Recall@k`, MRR, doc-route accuracy, entity/year coverage và latency. Không
chọn dense/reranker chỉ vì điểm benchmark tiếng Anh/tiếng Việt ngoài miền cao.
