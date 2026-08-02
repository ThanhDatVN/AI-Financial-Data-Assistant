# Nhật ký thí nghiệm và submission

Mỗi run phải ghi code/config/model revision/input hash/output hash/thời gian. Số đo chưa có gold phải
được gọi đúng tên health check, không ghi thành Accuracy/Recall.

## Các run đã thực hiện

### RUN-001 — 02/08/2026 — full dataset audit

- code base: `5b65a24` + worktree đang phát triển;
- input: `AIGuruTinix/ViFinQA`, 1.973 report, 1.012 question;
- output mới: `data/interim/dataset_audit.json` (5.098 bytes, schema v1);
- rerun sau sửa NLU: 165,119 giây; SHA-256
  `e38428c1099e8cfa99498bd4e029e142dc95a4f19dad2d1280e859754b524602`;
- corpus: 146.246 opening = closing = balanced `<table>`;
- questions: ID unique/sequential; target unit `UNKNOWN=0` sau sửa longest-unit matching;
- entity: 1 câu global không nêu ticker được giữ unresolved thay vì gán giả;
- scope: consolidated 957, separate 954, aggregated 7, unknown 55 reports;
- kết luận: dataset card/companion metadata 143.815 là stale với release này.
- trạng thái: superseded bởi RUN-008 sau sửa temporal/entity NLU; giữ lại để truy vết lịch sử.

### RUN-002 — 02/08/2026 — full manifest

- command: `scripts/10_build_manifest.py`, 6 workers;
- thời gian: 581,687 giây;
- output JSONL: 540.304.571 bytes;
- output Parquet: 67.307.387 bytes;
- records/unique refs: 146.246/146.246;
- documents with tables: 1.965; empty/malformed: 0/0;
- unit view: VND 97.607; million VND 29.842; percent 7.414; unknown 6.596; shares 3.586;
  thousand VND 928; billion VND 273;
- cảnh báo: `table_ref` dùng companion grammar nhưng `dashboard_verified=false`.
- trạng thái: superseded bởi RUN-006 sau khi phát hiện context/unit của bảng trước có thể rò sang bảng
  sau; không dùng hash/phân bố unit RUN-002 cho run mới.

### RUN-003 — 02/08/2026 — BM25 và metadata routing

- BM25 records: 146.246;
- build time: 81,2 giây;
- full query rows: 1.012; empty fused candidates: 0;
- multi-entity questions theo resolver: 320;
- lỗi phát hiện: HND bị mất route do scope quá chặt; TKV cần alias DTK; merge ban đầu để một entity lấn
  entity khác;
- sửa: scope policy nhận target + unknown, alias/word-boundary, per-ticker route và balanced round-robin;
- lưu ý: file full hiện có được tạo trước lần sửa balance cuối; RUN-004 phải tái sinh trước khi dùng.

### RUN-004 — 02/08/2026 — final metadata-prefiltered balanced CPU retrieval

- command: `scripts/30_retrieve_questions.py --candidate-k 2000`;
- output: `outputs/retrieval.jsonl`, 1.012 ID duy nhất/liên tiếp;
- wall time: 52,009 giây;
- SHA-256: `a7d9d775b5397350c03d4403a08159a04e7f19c236a78b9dff58ed3458582a02`;
- empty / ref ngoài manifest / ref sai metadata route: 0 / 0 / 0;
- available `ticker × report_year` route missing: 0; max route/question: 18;
- multi-entity questions sau entity-role filter: 280;
- fallback sang report năm kế tiếp do không có report đúng năm: PDR-2024, BSR-2016, BID-2021;
- lỗi đã sửa trong run: post-filter top-2.000 làm rơi route; chỉ balance ticker mà không balance năm;
  ticker nằm trong alias dài; counterparty bị coi là primary; cohort bị collapse bởi heuristic quá rộng.
- trạng thái: superseded bởi RUN-007 sau rebuild parser/manifest và sửa temporal/entity routing.

### RUN-005 — pending external GPU

- BGE-M3 dense + optional reranker + Qwen/vLLM;
- phải pin revision SHA trước run final;
- chạy smoke 5 → inspect → full 1.012; log `pip freeze`, GPU, throughput, schema/execution errors.

### RUN-006 — 02/08/2026 — strict-context manifest v2

- lỗi gốc: `context_before` có thể chứa HTML bảng liền trước, làm unit `%`/VND bị gán sang bảng sau;
- sửa: `</table>` là hard boundary, numbered section được carry riêng; unit evidence override theo
  column rồi row, thêm USD/million USD;
- command: `scripts/10_build_manifest.py --workers 6`; 303,0 giây;
- records/unique refs: 146.246/146.246; 1.965 source paths; zero row/column: 0/0;
- JSONL: 426.422.713 byte, SHA-256
  `ced1d671d6a71c299fea02d7d12b86b596b430a2714ab9aeaa4b338fe012fac1`;
- Parquet: 50.139.191 byte, SHA-256
  `060bd26eff14d30ce70b3ba7b00af509be6100b58ddb5a0fe970afa0ef69e29d`;
- table-default unit: VND 74.133; million VND 29.584; unknown 29.068; percent 7.902; shares
  3.744; thousand VND 942; USD 509; billion VND 349; million USD 15;
- real regressions: VJC q1 = 208253,201298 triệu VND; ACV q213 = 6,15569834 triệu USD;
- contract vẫn `dashboard_verified=false`.

### RUN-007 — 02/08/2026 — BM25/retrieval v2 + fail-closed QC

- BM25 rebuild: 146.246 records, 31,4 giây, artefact 170.777.097 byte;
- retrieval: 1.012 rows, 54,9 giây, SHA-256
  `99bcd3646952d20afc5fa201c9308a05b75c800fac7399287ec34d9d0538bf09`;
- QC report SHA-256
  `eadf1c9e409b06d9b197ebd098af45cff68fd5c6f7345c84713e1e27cc2e62b6`;
- empty / duplicate / ref ngoài manifest / ref sai metadata / available route missing: 0/0/0/0/0;
- 289 multi-entity questions; max 18 route/question; max 20 candidate/question;
- 3 route không có report đúng metadata: PDR-2024, BSR-2016, BID-2021;
- lỗi NLU sửa thêm: `Hòa Phát đạt` không còn sinh PDR; `ngành công nghiệp` không còn sinh SNZ;
  inclusive year range được expand; 10 câu so sánh nhiều `công ty mẹ` không còn co về entity đầu.

### RUN-008 — 02/08/2026 — audit v2 sau parser/NLU

- command: `scripts/00_audit_dataset.py`; 398,0 giây;
- SHA-256: `de99bf5db315f84330a06d1d97e29d1f9debf9eac7810a3c1f153990279782b0`;
- target unit unknown 0; entity unresolved duy nhất q464; 289 multi-entity và 360 multi-year;
- corpus invariants giữ nguyên: 1.973 docs, 146.246 balanced tables, 0 empty/malformed sample.

### RUN-009 — 02/08/2026 — deterministic unlabeled qrels template

- command: `scripts/60_sample_qrels.py --size 100`, seed 20260802;
- 100 unique IDs trên 77 strata; JSON Schema valid; mọi status `unlabeled`; 0 gold field có dữ liệu;
- hai lần chạy liên tiếp cùng SHA-256
  `a43f806516572bb6b1a7b03206a55d7f82cb7537c40a9f737b7bab03dc901e53`;
- không dùng artefact này để báo metric trước gán nhãn kép/adjudication.

### RUN-010 — 02/08/2026 — structured IR và isolated execution local gate

- model output contract đổi từ raw Pandas sang recursive typed IR JSON Schema;
- parser từ chối field thừa/depth/operator sai; grounding kiểm selected variables, coordinate duy nhất,
  source unit, value column, dimension và target divisor trước deterministic compile;
- execution/re-execution dùng child process `spawn` với timeout; POSIX hỗ trợ memory limit tùy chọn;
- OS-level no-network/read-only filesystem/CPU quota và panel generation vẫn pending;
- local hardware check: RTX 3050 4 GiB/WDDM hiện diện nhưng project env là torch CPU,
  `cuda_available=false`; không chạy vLLM Windows/native;
- chưa chạy Qwen, nên RUN-010 chỉ là code/local-test gate, không có Answer/Execution Accuracy;
- quality result: 52 pytest passed; Ruff/format clean; mypy strict 52 source files; compileall clean.

### RUN-011 — 02/08/2026 — deep-research expansion và qrels v2/pooling scaffold

- đối chiếu companion commit `9a046de`: 9 scenario trung gian, 3 formula bật và 70 hard grammar;
  release không có mapping câu→tier/template nên không gán difficulty theo ID/từ khóa;
- mở rộng reference pack tới file 00–12: retrieval, subtable, financial reasoning, verification,
  no-gold/calibration, XBRL/IFRS semantics, synthetic supervision và execution security;
- qrels schema v2 thêm grouped fingerprint, answerability states, cell/intermediate/cohort traces,
  operator/formula, unit/tolerance/rounding và double-annotation/adjudication roles;
- template: 100 unique ID/77 strata/all unlabeled, SHA-256
  `1584fe25a4ba7889d822f93ecaa8916e00fe57f01700172f09ffbda7136a0114`;
- pooling tool smoke ban đầu dùng artefact cũ `retrieval_balanced.jsonl`, tạo 1.962 candidates;
  RUN-012 phát hiện file này không phải frozen retrieval v2 và đã supersede pool/hash tương ứng;
- pool chỉ từ BM25/fused cùng run nên vẫn không phải diverse pool/qrels/metric;
- quality: 56 pytest; Ruff/format 71 files; mypy strict 53 source files; compileall/pip check xanh;
  2 notebooks/14 code cells compile; BibTeX 47 unique keys; 0 local link thiếu trong project docs.

### RUN-012 — 02/08/2026 — local final preflight + dense compatibility smoke

- kiểm lại frozen artefacts: audit SHA `de99bf5d...782b0`; manifest JSONL/Parquet giữ SHA
  `ced1d671...fac1` / `060bd26e...e29d`; BM25 đủ 146.246 records;
- phát hiện `outputs/retrieval_balanced.jsonl` là artefact cũ SHA `613419a0...`, không phải frozen v2;
  frozen control đúng là `outputs/retrieval.jsonl` SHA `99bcd364...38bf09`;
- BM25 full rerun trong thư mục riêng: 1.012 câu, 55,561 giây, byte-identical frozen v2;
  QC SHA `eadf1c9e...2e62b6`, `passed=true`, mọi failure count bằng 0;
- qrels control pool được tái sinh từ frozen v2: 100 câu/1.998 candidates, SHA
  `8d3519bec6dcc6baf08ba4b1d01f8e445e72c0140c2f59140fe2544049ac4eba`;
- BGE-M3 CPU/offline smoke: revision `5617a9f61b028005a4858fdac845db406aefb181`, 100 bảng,
  147,3 giây; positive q471 trả 5 dense hit đúng route `AAA–2016–consolidated`;
- Hugging Face API xác minh full immutable revisions: dense `5617a9f...` (03/07/2024), reranker
  `953dc6f...` (24/06/2024), Qwen AWQ `8e8ed24...` (18/11/2024), đều trước cutoff;
- notebook 02 thêm pinned-config cell, full opt-in mặc định tắt và TP=1; handoff manifest SHA
  `fef7b2cb25a2d8c8bd718110cffd954b8a1ed2c40ea0b3c4e8136092b3e63a4b`, hash/size check 0 lỗi;
  notebook được harden thêm: config rerun ghi đè đúng cờ, input hash gate, dense/retrieval assertions,
  smoke gate `5/0/5` và full ZIP hash;
- quality cuối: 56 pytest; Ruff/format 71 files; mypy strict 53 files; compileall/pip check xanh;
  2 notebooks/15 code cells compile; 27 Markdown files có 0 local link thiếu; 47 BibTeX keys unique;
- chưa chạy: full dense CUDA, reranker runner, Qwen/vLLM, human qrels, dashboard contract/submission.

## Quality gates gần nhất

| Gate | Kết quả đã ghi nhận | Cần chạy lại cuối phiên |
|---|---:|---|
| pytest | 56 passed | có |
| Ruff | all checks passed | có |
| mypy strict | success, 53 source files | có |
| pip check | no broken requirements | có |
| real-corpus q1 | 208253,201298 triệu VND | regression test |
| real-corpus q213 | 6,15569834 triệu USD | regression test |
| retrieval validator | `passed=true`, 0 failure | chạy sau mỗi retrieval |

## Mẫu run mới

```text
### RUN-<nnn> — <timestamp UTC+07>
- commit/worktree:
- config + sha256:
- input dataset/revision/hash:
- models: base/repo@revision, params, license, quantization:
- environment: Python/torch/CUDA/vLLM/GPU:
- command/seed:
- wall time + peak RAM/VRAM:
- artefacts + sha256:
- offline labels/metric definition:
- results by slice:
- failures and decision:
```

## Mẫu submission

```text
### SUB-<nnn> — <timestamp UTC+07> — public/private
- source RUN:
- ZIP sha256 + clean-room validation log:
- dashboard contract version/screenshot:
- purpose:
- score: F2 / Answer Accuracy / Execution Accuracy:
- quota shown before/after:
- keep/reject decision:
```

Không tự ghi quota “10/ngày, private 5” nếu dashboard hiện tại hiển thị khác; đặc tả local là nguồn
ban đầu, dashboard tại thời điểm nộp là điểm cần đối chiếu.
