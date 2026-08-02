# Tài liệu tham khảo và hồ sơ kiểm chứng

Thư mục này chỉ giữ các nguồn có thể truy vết về tài liệu gốc: đặc tả/dataset chính thức, paper
arXiv/ACL, model card của nhà phát hành và tài liệu thư viện chính thức. Blog hoặc kết quả tìm kiếm
chỉ được dùng để tìm nguồn, không được dùng làm bằng chứng trong working notes paper.

**Lần rà soát:** 02/08/2026. “Đã kiểm chứng” ở đây nghĩa là đã đối chiếu metadata, abstract/model
card/API và phạm vi áp dụng; không đồng nghĩa đã tái lập toàn bộ thí nghiệm của paper.

## Danh mục

| File | Nội dung | Kết quả dùng trong hệ thống |
|---|---|---|
| [00-research-matrix.md](00-research-matrix.md) | Ma trận nguồn → kết luận → code | Điểm vào chính |
| [01-text-to-sql-pandas.md](01-text-to-sql-pandas.md) | Text-to-Pandas, executable code | IR nhỏ + thực thi/kiểm chứng |
| [02-table-retrieval.md](02-table-retrieval.md) | Truy hồi bảng | metadata route + BM25/dense/RRF |
| [03-financial-table-qa.md](03-financial-table-qa.md) | Financial numerical QA | evidence-grounded programs |
| [04-vietnamese-nlp.md](04-vietnamese-nlp.md) | IR/embedding tiếng Việt | BM25 bỏ dấu + BGE-M3 ablation |
| [05-table-extraction-ocr.md](05-table-extraction-ocr.md) | Parser và provenance | parse HTML đã OCR, không OCR lại |
| [06-mo-hinh-va-inference.md](06-mo-hinh-va-inference.md) | Model/runtime/compliance | Qwen 7B AWQ + vLLM hiện hành |
| [07-bai-toan-taxonomy.md](07-bai-toan-taxonomy.md) | Taxonomy evidence/operator/temporal/population | Coverage contract, không gán tier giả |
| [08-ky-thuat-moi-va-ablation.md](08-ky-thuat-moi-va-ablation.md) | Research 2020–2026 và kỹ thuật mới | Ma trận applied/next/ablation/deferred |
| [09-evaluation-calibration-no-gold.md](09-evaluation-calibration-no-gold.md) | Qrels, leakage, calibration, abstention | Protocol khi release không có gold |
| [10-financial-semantics.md](10-financial-semantics.md) | XBRL/IFRS, time/scope/unit/formula | Typed fact + metric registry |
| [11-synthetic-data-training.md](11-synthetic-data-training.md) | Program-first synthetic data, negatives, QLoRA | Chỉ sau split/qrels và executable validation |
| [12-security-sandbox-provenance.md](12-security-sandbox-provenance.md) | Threat model, DSL, isolation, hosted runtime | Không gọi executor hiện tại là adversarial sandbox |
| [bibliography.bib](bibliography.bib) | 47 BibTeX entry đã lọc | 47 key duy nhất; không có entry `unverified` |

## Trạng thái bằng chứng

| Nhãn | Nghĩa |
|---|---|
| `verified` | Đã mở nguồn gốc và kiểm metadata/claim đang sử dụng |
| `applied` | Đã ánh xạ vào module/config/test cụ thể |
| `ablation` | Có cơ sở để thử nhưng chưa có kết quả trên ViFinQA |
| `deferred` | Hợp lý nhưng chưa đủ dữ liệu/phần cứng hoặc không ở critical path |
| `rejected` | Không phù hợp rule, domain hoặc chi phí |
| `unresolved` | Chỉ BTC/dashboard mới có thể xác nhận |

Không ghi số benchmark ngoài miền như thể đó là hiệu quả dự kiến trên ViFinQA. Release công khai không
có qrels/gold/dev split; mọi Recall/F2/Answer/Execution Accuracy chỉ được ghi khi có log dashboard hoặc
gold do nhóm tạo và công bố cách sinh.

## Nguồn dữ liệu đang dùng

| Nguồn | Vai trò | Trạng thái |
|---|---|---|
| [ViFinQA](https://huggingface.co/datasets/AIGuruTinix/ViFinQA) | corpus và 1.012 câu hỏi | `applied`; CC BY-NC 4.0 |
| [Companion ViFinQA](https://github.com/DSKT-NOWJ/ViFinQA) | đối chiếu parser/ref/eval | `verified` tại commit `9a046de` |
| Văn bản thể lệ trong repo | schema/metric/rule/calendar | `verified local`; phải kiểm dashboard trước nộp |

Hiện pipeline **không nhập thêm dữ liệu số tài chính bên ngoài**. Các văn bản kế toán chỉ là nguồn
tham khảo tên chỉ tiêu/scope/unit; không được dùng để thay số liệu corpus:

- [TT200/2014/TT-BTC — lịch sử hiệu lực](https://vbpl.vn/TW/Pages/vbpq-lichsu.aspx?ItemID=66801):
  hữu ích cho corpus lịch sử nhưng hết hiệu lực từ 01/01/2026 khi TT99/2025 thay thế.
- [TT49/2014/TT-NHNN](https://vbpl.vn/nganhangnhanuoc/Pages/vbpq-thuoctinh.aspx?ItemID=52646):
  tham chiếu biểu mẫu tổ chức tín dụng.
- [TT334/2016/TT-BTC](https://vbpl.vn/TW/Pages/vbpq-thuoctinh.aspx?ItemID=118869):
  tham chiếu hệ thống tài khoản/chỉ tiêu liên quan công ty chứng khoán.

## Quy tắc đọc research pack

- `applied` chỉ có nghĩa kỹ thuật đã nối vào code/test, không có nghĩa đã đạt accuracy.
- `next` là critical path đã có evidence local và nguồn sơ cấp hỗ trợ.
- `ablation` phải có qrels/dev và run thay đúng một biến trước khi promote.
- `deferred` không đồng nghĩa kỹ thuật kém; nó không khớp dữ liệu/compute/rule hiện tại.
- Mọi paper 2026 đã được đối chiếu trang proceedings/ACL chính thức tại ngày rà soát; preprint gần
  thời điểm cutoff chỉ được dùng như giả thuyết và phải ghi rõ `preprint` nếu đưa vào experiment.

Taxonomy trong file 07 dựa trên code companion để lập coverage, nhưng release không có template ID.
Không được báo tỷ lệ tier/operator như ground truth từ count từ khóa hoặc vị trí question ID.

## Quy tắc đưa nguồn vào paper

1. Dùng URL/DOI/arXiv/ACL gốc; ghi ngày truy cập với nguồn động.
2. Claim định lượng phải chỉ rõ dataset/setting của paper, không ngoại suy sang ViFinQA.
3. Model phải ghi cả base model, repo lượng tử hóa thực dùng, revision SHA, số tham số, license và
   release date.
4. Dữ liệu ngoài phải ghi nguồn, license, checksum, script thu thập và câu hỏi nào chịu ảnh hưởng.
5. Bất kỳ điểm nào chưa rõ từ dashboard (table-ref grammar, tolerance, quota) giữ nhãn `unresolved`.
