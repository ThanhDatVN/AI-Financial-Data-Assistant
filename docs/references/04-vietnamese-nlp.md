# 04 — Truy hồi và chuẩn hóa tiếng Việt

## 1. Nguồn đã kiểm chứng

| Nguồn | Phạm vi | Cách dùng đúng |
|---|---|---|
| [VN-MTEB](https://arxiv.org/abs/2507.21500), Pham et al. 2025 | 41 dataset, 6 nhóm tác vụ embedding tiếng Việt | shortlist model; không thay evaluation tài chính |
| [Vietnamese IR benchmark](https://arxiv.org/abs/2503.07470), Nguyen et al. 2025 | retrieval/reranking tiếng Việt và learning objective | thiết kế error slice/ablation |
| [BGE-M3 paper](https://arxiv.org/abs/2402.03216) và [card](https://huggingface.co/BAAI/bge-m3) | multilingual, dense/sparse/multi-vector, long context | dense default để thử |
| [BGE reranker card](https://huggingface.co/BAAI/bge-reranker-v2-m3) | multilingual cross-encoder | optional rerank |

Điểm VN-MTEB không phải điểm ViFinQA. Model chỉ được chốt sau Recall@k/latency/memory trên bộ qrels
nội bộ hoặc log dashboard hợp lệ.

## 2. Hai biểu diễn lexical song song

Tiếng Việt OCR có lỗi dấu, Unicode composition và khoảng trắng. Mỗi field lưu:

- bản NFC, giữ dấu để bảo toàn nghĩa;
- bản `label_ascii`, bỏ dấu/lower/collapse whitespace để match robust;
- raw text + hash/offset để quay lại nguồn;
- token không chữ số cho semantic retrieval.

Không chỉ dùng bản bỏ dấu vì có thể tăng collision. BM25 tìm trên cả hai view; fuzzy chỉ tie-break sau
metadata route.

## 3. Word segmentation

Word segmentation không được bật mặc định cho mọi encoder. BGE-M3 dùng tokenizer của chính model.
PhoBERT-based encoders có thể cần word segmentation theo model card, nhưng OCR và thuật ngữ/mã chứng
khoán khiến segmentation dễ tạo lỗi. Mọi preprocessing phải nằm trong model-specific adapter và được
ghi vào config/hash.

`unicodedata.normalize("NFC", text)` chạy trước normalization khác. Alias phải có word boundary/longest
match; regression `thuần bình quân` bảo đảm chuỗi thường không bị nhận nhầm thành ticker ABB.

## 4. Baseline và ablation

| Tầng | Baseline | Ablation | Gate |
|---|---|---|---|
| document route | ticker/company alias + year/scope rules | LLM NLU | entity/year accuracy |
| sparse table | BM25 NFC + accentless | field weights | Recall@k/latency |
| dense table | none ở CPU baseline | BGE-M3 | incremental Recall@k |
| rerank | none | bge-reranker-v2-m3 | gain vs latency/VRAM |
| row linking | normalized exact/fuzzy | dense label | gold-cell accuracy |

BGE-M3/reranker là `pending GPU`, không ghi “model tốt nhất” trước ablation. Các candidate Halong hoặc
Vietnamese bi-encoder có thể thêm sau khi kiểm license/revision/preprocessing, nhưng không cần đưa vào
baseline khi chưa có qrels.

## 5. Slice kiểm thử bắt buộc

- có dấu vs bỏ dấu;
- ticker trùng từ thường;
- tên pháp nhân cũ/mới/brand/viết tắt;
- ngân hàng, chứng khoán, doanh nghiệp thường;
- câu đa ticker;
- từ đồng nghĩa chỉ tiêu (`lãi sau thuế`, `LNST`, `lợi nhuận sau thuế`);
- lỗi OCR tách/dính ký tự;
- query global không nêu ticker (không ép resolve giả).
