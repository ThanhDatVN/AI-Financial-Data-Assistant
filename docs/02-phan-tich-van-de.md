# Phân tích vấn đề và thiết kế hệ thống

## 1. Phân rã lỗi end-to-end

Một answer sai có thể đến từ nhiều lớp độc lập:

| Lớp | Failure mode chính | Cách phát hiện |
|---|---|---|
| corpus | thiếu/sai phiên bản file | checksum, audit count |
| identity | lệch 0/1-based, `doc|N`/`doc|table_N` | contract-test dashboard |
| parse | rowspan/colspan/header/unit sai | structural test + manual sample |
| NLU | thiếu ticker, kỳ, scope, target unit | audit 1.012 QuerySpec |
| retrieval | không lấy đủ bảng hoặc một entity lấn entity khác | recall@k, per-route coverage |
| selection | đúng bảng nhưng sai row/column | cell-coordinate audit |
| program | sai công thức, zero divide, wrong aggregation | execution + metamorphic tests |
| unit | nhầm nghìn/triệu/tỷ, %/ratio | unit lineage assertion |
| package | thiếu ID/CSV, path sai, query-answer lệch | offline validator |

Không nên dùng một LLM end-to-end để “tự sửa” tất cả. Cần log lỗi theo taxonomy và cải thiện đúng
tầng gây lỗi.

## 2. Parsing OCR table

Corpus chứa HTML inline tương đối sạch về cặp tag nhưng nội dung OCR vẫn nhiễu. Parser hiện:

- tách bảng bằng regex cân bằng và giữ page/line/char offset;
- mở rộng rowspan/colspan bằng occupancy map;
- chuẩn hoá NFC/space nhưng giữ raw HTML và SHA-256;
- infer tối đa ba header row theo tỷ lệ numeric/text;
- deduplicate header;
- lưu context trước bảng và section title;
- parse số Việt Nam thận trọng: dấu chấm hàng nghìn, phẩy thập phân, ngoặc âm, dash;
- từ chối chuỗi OCR mơ hồ như `1O0.000`.

Các điểm chưa hoàn thiện về mặt nghiên cứu:

- header inference heuristic có thể sai với bảng toàn text;
- unit anchor xa hơn 8 dòng hoặc nằm sau bảng có thể bị bỏ;
- bảng continuation nhiều trang chưa được merge;
- OCR hỏng label cần fuzzy/semantic canonicalization nhưng không được sửa raw cell;
- một số báo cáo không ghi rõ scope trong tên file.

Do public release đã có inline HTML, OCR hình ảnh/PDF không nằm trên critical path. Chỉ thêm OCR
vision khi BTC cấp file ảnh khác với corpus này.

## 3. Query understanding

`QuerySpec` tách:

- entity/ticker và confidence;
- scope explicit (`separate`, `consolidated`) hoặc unspecified;
- temporal mentions và column role;
- target unit/divisor;
- câu gốc.

Company resolver dùng ba tầng: ticker exact; unique word-bounded alias/brand; fuzzy fallback chỉ khi
điểm cao và có margin. Word boundary là bắt buộc: compact substring từng làm `an binh` khớp nhầm
qua cụm `thuần bình quân`.

Policy routing:

- ticker confidence cao được dùng làm hard filter;
- scope explicit tìm scope đó **và** tài liệu `unknown`, không tìm scope đối nghịch;
- scope không explicit không hard-filter;
- multi-entity chạy route riêng rồi round-robin;
- nếu câu hỏi thật sự là toàn corpus, không ép ticker.

## 4. Table representation

Một bảng có nhiều “view” với mục tiêu khác nhau:

1. metadata view: ticker, year, scope, section;
2. schema view: headers;
3. row-label view: label text, bỏ numeric value;
4. context view: vài dòng trước bảng;
5. optional synthetic-question/semantic-label view;
6. execution view: long CSV có raw/numeric/base value.

Không dùng cùng một linearization cho retrieval và execution. Retrieval không cần toàn bộ số; executor
cần số và coordinate chính xác.

## 5. Retrieval architecture

### 5.1 Baseline đã chạy

BM25 dùng original + accentless text, metadata route và top-k. Nó rẻ, giải thích được và rất mạnh với
financial labels lặp lại. Full index 146.246 bảng đã build thành công.

### 5.2 Dense + reranker

Candidate GPU:

- BGE-M3 cho dense embedding; model card hỗ trợ hơn 100 ngôn ngữ, dense/sparse/multi-vector và
  context đến 8.192 token;
- bge-reranker-v2-m3 cho cross-encoder multilingual;
- FAISS inner product trên vector normalized;
- RRF để fusion sparse/dense trước rerank.

Không coi benchmark Vietnamese tổng quát là bằng chứng chắc chắn cho financial table retrieval. Phải
đo trên qrels ViFinQA khi có.

### 5.3 Multi-view và adaptive k

Paper THYME chỉ ra field của table có matching preference khác nhau; QGpT tăng alignment bằng câu
hỏi tổng hợp từ partial table; DCTR 2026 tách query có kiểu và xét connectivity; Adaptive Table
Retrieval 2026 thay fixed-k bằng threshold/sliding rerank; T-RAG 2026 dùng hierarchical memory và
multi-stage retrieval. Những ý này chuyển thành ablation, không được đưa thẳng vào production:

- `A0`: metadata BM25;
- `A1`: BM25 row-label/schema;
- `A2`: BGE-M3 dense;
- `A3`: BM25+dense RRF;
- `A4`: A3 + reranker;
- `A5`: multi-view row chunks/semantic labels;
- `A6`: adaptive stopping theo calibrated marginal score.

Gate chọn phương pháp là macro F2/recall@k trên dev qrels và downstream execution accuracy, không
phải chỉ nDCG của paper khác domain.

## 6. Cell selection và Text-to-Pandas

DataFrame evidence dạng long:

| Cột | Ý nghĩa |
|---|---|
| table_ref/doc_id/ticker/year/scope | provenance |
| row_index/column_index | coordinate ổn định trong bảng đã parse |
| row_label/column_label | schema cho model/người đọc |
| raw_value | OCR nguyên gốc |
| numeric_value | số trong đơn vị nguồn |
| base_value | số sau source multiplier |
| source_unit/is_dash/is_missing | lineage/QC |

Prompt program chỉ đưa metadata, label và coordinate, không đưa source value. Model chọn variable/cell
và tạo một expression. Executor mới đọc CSV và tính số. Cách này giảm nguy cơ model chép nhầm số,
nhưng không tự giải quyết câu phức tạp cần argmax/filter theo giá trị. Lộ trình:

1. scalar cell lookup và arithmetic;
2. typed IR cho aggregate/comparison/rank;
3. panel IR cho entity × year;
4. code model chỉ sinh IR JSON;
5. deterministic compiler sinh pandas expression;
6. differential execution so với NumPy/reference operator.

Generation hiện sinh typed IR theo JSON Schema; parser/grounding checker kiểm field, coordinate,
selected variable, source unit và dimension trước khi compiler tạo Pandas. Kết nối đã có code/test
nhưng chưa có log Qwen GPU, nên chưa được claim đạt hard tier; panel coverage vẫn còn thiếu.

## 7. Sandbox và an toàn thực thi

Executor hiện parse AST ở mode expression và chặn:

- statement/import/lambda/comprehension;
- tên ngoài evidence variables và allowlisted scalar functions;
- private/dunder attributes;
- method calls ngoài allowlist;
- non-scalar, boolean, NaN/Inf result.

Execution hiện chạy trong process `spawn` riêng và bị terminate theo timeout; POSIX có tùy chọn
`RLIMIT_AS`. AST vẫn là language guard. Read-only filesystem, OS-level network denial, CPU quota và
memory isolation tương đương trên Windows chưa có, nên chưa gọi đây là sandbox chống đối thủ hoàn chỉnh.

## 8. Đơn vị và số học

Unit là type, không phải comment. Mỗi operator cần rule:

- cộng/trừ: hai operand cùng currency/base dimension;
- nhân/chia: dimension mới (ratio, currency²...) phải hợp lệ;
- percentage: giữ percentage points nếu câu hỏi hỏi %;
- mean/median: không bỏ NaN/dash âm thầm;
- growth: định nghĩa `(new-old)/abs(old)` hay `/old` phải theo đề;
- division by zero: fail có trace, không trả 0;
- rounding: chỉ ở output nếu câu hỏi/BTC quy định.

TT200/2014, TT49/2014, TT334/2016 giúp hiểu schema lịch sử của corpus, nhưng không dùng văn bản pháp
lý để “sửa” con số OCR. TT200 đã được TT99/2025 thay từ 2026; corpus lịch sử vẫn theo chế độ tại
thời điểm báo cáo.

## 9. Đánh giá khi chưa có gold

Có thể chạy:

- parser invariants trên toàn corpus;
- NLU coverage;
- metadata route consistency;
- query re-execution;
- unit/type assertions;
- manual stratified audit;
- synthetic/metamorphic programs;
- q1 integration smoke.

Không được gọi các proxy trên là retrieval/answer accuracy chính thức. Khi có qrels:

- report/table precision, recall, macro F2 tại k;
- hit/recall@1,3,5,10,20,50,100 để parity companion;
- answer exact/tolerance accuracy;
- execution accuracy;
- error slice theo scope, unit, #entity, #year, operator, OCR severity;
- bootstrap confidence interval và paired significance cho ablation.

## 10. Rủi ro ưu tiên

| Rủi ro | Xác suất | Tác động | Mitigation |
|---|---|---|---|
| dashboard table-ref khác companion | cao | rất cao | contract-test public sớm |
| không có qrels/dev | cao | rất cao | xin BTC; manual stratified set; không overclaim |
| unit sai | trung bình | rất cao | lineage + typed checks + abstain |
| multi-entity thiếu route | trung bình | cao | per-entity route + round-robin |
| model sinh query chạy được nhưng sai | cao | cao | typed IR, execute, metamorphic tests |
| corpus/model revision drift | trung bình | cao | hashes + immutable revisions |
| GPU/runtime thay đổi | cao | trung bình | detect hardware + checkpoint + CPU fallback |
| private submission budget bị lãng phí | trung bình | rất cao | freeze configs; no probing private |

## 11. Taxonomy mở rộng và sửa giả định

Không được đồng nhất độ khó với số phép tính hoặc question ID. Release không có tier/template label;
mọi thống kê theo từ khóa chỉ là lexical slice và có overlap. Đối chiếu code companion cho thấy 9
scenario trung gian và 70 hard grammar, nhưng không có mapping trở lại từng câu release. Plan vì thế
đổi từ “dự đoán tier” sang coverage theo năm trục:

| Trục | Các trạng thái cần model hóa |
|---|---|
| evidence | cell → table → report → multi-report → multi-company |
| time | instant/duration, two-period, time series, next-period dependency |
| population | entity, peer group, filtered cohort, global |
| logic | lookup, arithmetic, aggregate, set/filter, rank, finance formula |
| answerability | sufficient, missing, retrieval incomplete, ambiguous/conflict, execution failure |

Các ngách high-risk gồm ratio-of-sums, cohort median/quantile, multi-predicate intersection, entity
được chọn bằng metric A rồi lookup metric B, sign transition, restated comparatives và formula dùng
average balance. Chi tiết/coverage gate ở
[taxonomy reference](references/07-bai-toan-taxonomy.md).

## 12. Quyết định kỹ thuật sau deep research

| Vấn đề local | Candidate | Quyết định |
|---|---|---|
| filings gần trùng nhau | hierarchical doc→table + finance constraint cards | `next`; đo route/evidence recall |
| exact term và paraphrase tiếng Việt | BM25, dense, SPLADE, late interaction, RRF | factorized ablation; BM25 luôn là control |
| bảng lớn/nhiều cell nhiễu | iterative subtable selection | gold-table ablation trước end-to-end |
| multi-hop formula | typed DSL + metric dependency graph | `next`; deterministic executor |
| program chạy được nhưng sai | cell/set/unit/formula/cross-path verifier | `next`; execution success chỉ là một gate |
| thiếu dữ liệu | explicit state + calibrator + abstention | `next`; report risk–coverage |
| global long-context/GraphRAG | hierarchical summary/graph RAG | `deferred`; không khớp exact-scalar critical path |
| PDF/image | table detector/structure/OCR/MLLM | `deferred`; corpus hiện đã có HTML |
| generated arbitrary Python | container/microVM sandbox | production `rejected`; DSL là đường chính |

FinCARDS/HiREC đặc biệt sát corpus vì entity/metric/period/scope giải quyết near-duplicate. PieTa,
SPLADE/ColBERT và graph cross-path verification đáng thử nhưng chỉ được promote sau qrels. Xem ma trận
nguồn và ablation tại [references/08](references/08-ky-thuat-moi-va-ablation.md).

## 13. Data/evaluation strategy mới

Ưu tiên số một không phải thêm model mà là tạo thước đo đáng tin:

1. qrels pool từ run lexical/dense/late/rerank + metadata routes;
2. hai annotator và adjudication cho report/table/cell/operator/formula;
3. grouped split theo entity/year/template/table fingerprint;
4. gold-table reasoning, retrieved-table deterministic và full generated program là ba track riêng;
5. risk–coverage cho abstention; không dùng LLM-as-judge thay numeric gold;
6. synthetic program-first chỉ trên train partition sau khi split.

Conformal calibration chỉ là ablation vì distribution shift có thể phá exchangeability. Human set 100
câu hiện là unlabeled sampling frame, không phải dev. Protocol đầy đủ ở
[references/09](references/09-evaluation-calibration-no-gold.md), semantic fact model ở
[references/10](references/10-financial-semantics.md), synthetic ở
[references/11](references/11-synthetic-data-training.md) và security ở
[references/12](references/12-security-sandbox-provenance.md).

## 14. Kiến trúc mục tiêu

```text
questions.jsonl
      │
      ▼
 deterministic NLU ────────────────┐
      │ QuerySpec                  │ metadata policy
      ▼                            ▼
 BM25 ─ dense / sparse / late interaction ─ route partitions
      └────────────────┬────────────────────┘
                       ▼
        RRF / finance-card rerank / entity balance
                 ▼
     candidate table schemas / compact subtables
                 ▼
       typed IR / metric dependency graph
                 ▼
       long evidence CSV + executor
                 ▼
 answer + intermediate trace + verifier/abstention
                 ▼
             submission.zip
```

Mỗi mũi tên phải có serializable artefact/checkpoint. Không để trạng thái quan trọng chỉ tồn tại trong
notebook memory.
