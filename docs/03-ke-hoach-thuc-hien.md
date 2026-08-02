# Kế hoạch thực hiện chi tiết và quality gates

**Mốc kế hoạch:** 02/08/2026.

**Trạng thái:** `DONE`, `READY-NOT-RUN`, `BLOCKED-EXTERNAL`, `TODO`.

## 1. Nguyên tắc vận hành

1. Không có metric thì không tối ưu bằng cảm giác.
2. Không có provenance thì prediction không hợp lệ nội bộ.
3. Không merge phase khi test/lint/type/QC phase đó chưa xanh.
4. Mọi run có config, code commit, data/model revision, seed, thời gian và artifact hash.
5. Public submission chỉ cho contract/ablation đã định trước; private chỉ dùng config đã freeze.
6. Không dùng model/API vi phạm `≤14B`, cutoff hoặc open-weight policy.

## 2. Work breakdown structure

### P0 — Contract, dữ liệu và research (`DONE`, trừ dashboard gate)

| ID | Việc | Deliverable | Gate |
|---|---|---|---|
| P0.1 | Đọc đặc tả và lập field contract | docs 00/01 | review chéo với source |
| P0.2 | Tải public release | `data/raw/ViFinQA` | 1.012 questions, 1.973 docs |
| P0.3 | Pin companion | commit `9a046de...` trong cache | ghi commit |
| P0.4 | Audit corpus | `dataset_audit.json` | counts/invariants |
| P0.5 | Research nguồn sơ cấp | references 00–12 + bibliography | URL/metadata/claim verified |
| P0.6 | Xác minh dashboard table-ref/tolerance | contract note | `BLOCKED-EXTERNAL` |
| P0.7 | Đối chiếu taxonomy generator | 9 scenario + 70 hard grammar coverage | không gán tier cho release |

### P1 — Parser và manifest (`DONE`)

| ID | Việc | Test/QC | Kết quả |
|---|---|---|---|
| P1.1 | segment inline table | page/line/offset/hash | 146.246 balanced |
| P1.2 | rowspan/colspan matrix | synthetic structural tests | pass |
| P1.3 | header inference | header unit tests/manual sample | pass baseline |
| P1.4 | number parser | locale/negative/dash/OCR refusal | pass |
| P1.5 | source/cell unit detection | full distribution + mixed-unit regression | 80,1% table default assigned; cell override USD/% |
| P1.6 | manifest JSONL/Parquet | unique ref/empty/malformed | 146.246 unique, 0 empty |
| P1.7 | parallel determinism | worker1 vs worker2 hash | byte-identical sample |

Lệnh tái sinh:

```powershell
.venv\Scripts\python.exe scripts\10_build_manifest.py --workers 6
```

### P2 — Query understanding (`DONE` baseline; `TODO` typed population)

| ID | Việc | Gate |
|---|---|---|
| P2.1 | target unit/divisor | 0 UNKNOWN trên 1.012 câu |
| P2.2 | scope explicit only | không default consolidated |
| P2.3 | temporal role/range | start/end/flow + inclusive `2020-2024` tests |
| P2.4 | entity alias/brand | chỉ câu global không có entity |
| P2.5 | word-bound alias | regression `thuần bình quân` ≠ An Bình |
| P2.6 | population intent/industry cohort | `TODO`: typed all/group/industry selector |
| P2.7 | LLM NLU fallback | `READY-NOT-RUN`, chỉ khi deterministic confidence thấp |
| P2.8 | entity role/counterparty | target vs `bán hàng với`/`phải trả cho`; cohort không bị collapse |
| P2.9 | short-alias collision | proper-name gate; `Hòa Phát đạt` không sinh PDR |
| P2.10 | answerability state | `TODO`: missing/retrieval/ambiguous/conflict/execution |
| P2.11 | temporal fact type | `TODO`: instant/duration/average-balance/restated |
| P2.12 | finance constraint card | `TODO`: entity/metric/period/scope/unit/predicate |

### P3 — Sparse retrieval (`DONE` baseline)

| ID | Việc | Gate |
|---|---|---|
| P3.1 | schema/context/row-label view | numeric source values không vào view |
| P3.2 | accent-tolerant BM25 | unit test accent + full index |
| P3.3 | safe metadata route | lọc ticker/năm/scope trước ranking; 0 empty trên 1.012 |
| P3.4 | multi-entity/year balance | `ticker × report_year` round-robin; 0 route khả dụng bị thiếu |
| P3.5 | export + fail-closed QC | JSONL + `retrieval_qc.json`; 0 failure |
| P3.6 | full run balanced policy | DONE: 54,9 s; SHA-256 `99bcd364...38bf09` |
| P3.7 | near-duplicate diagnostic | TODO: same header/label khác entity-year-scope |
| P3.8 | hierarchical doc→table route | TODO ablation; giữ exact metadata recall |
| P3.9 | finance-card constraint score | TODO ablation; soft score trước hard gate |

### P4 — Dense, fusion, rerank (`READY-NOT-RUN`, cần GPU)

| ID | Việc | Deliverable/Gate |
|---|---|---|
| P4.1 | BGE-M3 encode | normalized FAISS index; dimension/count check |
| P4.2 | dense retrieval | exact metadata-prefilter đã code; cần full GPU run |
| P4.3 | RRF | deterministic tie-break test |
| P4.4 | bge reranker | rerank depth 100/200/500 ablation |
| P4.5 | multi-view | row chunk + metadata + label ablation |
| P4.6 | evaluate | macro F2/recall and downstream exec, khi có qrels |
| P4.7 | SPLADE/learned sparse | optional GPU build; compare storage/CPU latency |
| P4.8 | multilingual late interaction | optional ColBERT candidate; token-level match |
| P4.9 | sparse→late cascade | profile recall/latency; không chạy full nếu R04 gate thua |
| P4.10 | subtable selection | gold-table row×column evidence accuracy trước E2E |

Chạy notebook Kaggle `02_kaggle_dense_and_generate.ipynb`; checkpoint dense index trước khi tải
LLM để nếu runtime reset không mất retrieval.

### P5 — Evidence, IR và execution (`PARTIAL`)

| ID | Việc | Trạng thái | Gate |
|---|---|---|---|
| P5.1 | long evidence CSV | DONE | raw/numeric/base/provenance |
| P5.2 | scalar typed IR/compiler | DONE | arithmetic unit tests |
| P5.3 | AST constrained executor | DONE | malicious syntax tests |
| P5.4 | actual q1 integration | DONE | 208253.201298 từ CSV |
| P5.5 | aggregate/count/rank IR | DONE baseline | mean/count-if/arg-extremum differential tests |
| P5.6 | panel entity×year gate | DONE baseline | dynamic route budget + selected-cell ticker/year coverage; semantic hard-panel GPU còn pending |
| P5.7 | dimension type checker | DONE baseline | incompatible sum/multiply/divide fail |
| P5.8 | process isolation/timeout | PARTIAL | spawn + timeout DONE; POSIX memory optional; network/read-only FS TODO |
| P5.9 | population/cohort IR | TODO | typed universe/filter/intersection membership |
| P5.10 | temporal/dependent lookup | TODO | identity giữ qua chọn entity và kỳ kế |
| P5.11 | formula registry | TODO | ROA/ROE/quick ratio + hard families, versioned |
| P5.12 | ratio-of-sums/quantile/tie | TODO | differential + exact cohort/set trace |
| P5.13 | cross-path/consistency verifier | TODO ablation | mismatch không tự sửa evidence |
| P5.14 | structured abstention | TODO | reason code, no forced numeric fallback |

Structured output đã nối vào typed IR parser → grounding/dimension gate → deterministic compiler →
isolated executor. Chưa có log Qwen GPU và chưa có panel coverage matrix, nên không suy diễn code/test
thành chất lượng end-to-end hard tier.

### P6 — LLM generation (`READY-NOT-RUN`, cần GPU/model download)

| ID | Việc | Gate |
|---|---|---|
| P6.1 | Qwen2.5-Coder-7B-Instruct-AWQ server | model/revision/license record |
| P6.2 | JSON schema output | 100% response parse trên smoke |
| P6.3 | schema-only prompt | source value không xuất hiện trong prompt test |
| P6.4 | 5-question smoke | no server/schema/execution error |
| P6.5 | 100-question stratified | failure taxonomy và manual review |
| P6.6 | full 1.012 checkpointed run | completed 1.012, no missing ID |
| P6.7 | one repair attempt ablation | chỉ giữ nếu tăng exec/answer dev rõ ràng |
| P6.8 | grammar-constrained decoding | JSON Schema/CFG vs parse-only latency/error ablation |
| P6.9 | program confidence features | route/constraint/grounding/unit/cross-path; không dùng raw logprob một mình |

Inference mặc định temperature 0, seed 20260802, không silent retry. Lỗi được ghi `errors.jsonl`.
Success lưu IR/compiled query vào `program_traces.jsonl`; fingerprint khóa schema, candidate depth,
max tokens, timeout, model/input revision.

### P7 — Evaluation, calibration và error analysis (`PARTIAL`; accuracy `BLOCKED-EXTERNAL`)

| ID | Việc | Deliverable/Gate |
|---|---|---|
| P7.1 | freeze grouped split | entity/year/template/table fingerprint; hash |
| P7.2 | build diverse retrieval pool | PARTIAL: BM25 v2 control 1.998 candidates; dense/late/rerank pending |
| P7.3 | double annotation | relevance + sufficient evidence + cells + rationale |
| P7.4 | adjudication | versioned qrels; disagreement log |
| P7.5 | retrieval tracks | sufficient-evidence Recall/F2@k + route/unjudged@k |
| P7.6 | reasoning upper bound | gold-table program + step/cell/set/formula/unit accuracy |
| P7.7 | end-to-end | answer accuracy + coverage + latency + error state |
| P7.8 | selective prediction | risk–coverage + calibration/Brier on answerability |
| P7.9 | statistical comparison | paired bootstrap/permutation + effect size + CI |
| P7.10 | slice regression | scope/entity/year/operator/OCR/answerability critical slices |

Manual set phải stratify theo năm trục taxonomy. Annotator lưu table/cell, ordered intermediate facts,
cohort, formula, unit và answerability reason. Template 100 câu đã sinh deterministic/schema-valid
nhưng vẫn là unlabeled sampling frame. Không gọi proxy route/execution là accuracy. LLM-as-judge chỉ
là diagnostic sau khi calibrate với human labels.

### P8 — Submission (`PARTIAL`; chưa được phép nộp tự động)

| ID | Việc | Trạng thái |
|---|---|---|
| P8.1 | Pydantic field validation | DONE |
| P8.2 | re-execute every query | DONE |
| P8.3 | ZIP root/path validator | DONE |
| P8.4 | full prediction | phụ thuộc P6 |
| P8.5 | table-ref contract test public | cần người vận hành/dashboard |
| P8.6 | freeze candidate configs | TODO |
| P8.7 | private submission | chỉ sau approval, tối đa theo luật |

Không upload dashboard từ script này nếu chưa có quyết định người dùng; thao tác submission là external
state change và dùng quota.

### P9 — Working notes paper (`READY`, chờ kết quả external)

Outline: task/data; corpus audit; parsing; QuerySpec; multi-view retrieval; grounded IR/execution;
experiments/ablation; error analysis; compliance; limitations; reproducibility. Mỗi bảng paper phải
truy được về run directory/config/hash. Khung điền có gate tại `docs/07-working-notes-paper-outline.md`.

### P10 — Synthetic supervision và security (`TODO`, sau P7)

| ID | Việc | Gate |
|---|---|---|
| P10.1 | program-first synthetic generator | train partition only; exact round-trip execution |
| P10.2 | controlled hard negatives | one auditable entity/year/scope/metric violation |
| P10.3 | counterfactual unanswerable | exhaustive corpus check; no false-unanswerable |
| P10.4 | QLoRA/adapter ablation | frozen dev gain + no critical-slice regression |
| P10.5 | DSL-only production path | raw generated Python disabled |
| P10.6 | hostile-code research sandbox | Linux deny-network/RO mount/quota/image digest nếu thật sự cần |
| P10.7 | supply-chain/hosted audit | model/package revision, hashes, secrets scan, checkpoint |

Không bắt đầu P10.1–P10.4 trước khi split/qrels được freeze. Executor Windows hiện chỉ partial isolation;
không quảng bá là sandbox chống đối thủ.

## 3. Kế hoạch theo ngày

Ngày thực tế phải điều chỉnh theo dashboard, nhưng thứ tự gate không đổi.

### 02–04/08

- hoàn tất docs/research/runbook;
- hỏi BTC và contract-test table ref/tolerance;
- giữ balanced BM25 full hiện có làm frozen control;
- tạo diverse pool rồi gán nhãn kép/adjudicate template 100 nếu BTC chưa cấp gold;
- freeze parser manifest v1.

### 05–09/08

- Kaggle build dense index;
- BM25/dense/RRF/rerank/finance-card factorized ablation;
- profile latency/VRAM/storage;
- chọn retrieval candidate depth;
- kiểm mỗi multi-entity route.
- chỉ thử SPLADE/late-interaction nếu qrels đủ và baseline gate xanh.

### 10–16/08

- hoàn thiện cohort/temporal/dependent-lookup/formula IR;
- tạo operator differential/metamorphic test;
- Qwen smoke → stratified → full run;
- phân loại mọi failure, không sửa prompt ad hoc theo từng câu test.
- thêm intermediate-fact và abstention traces.

### 17–22/08

- end-to-end ablation;
- tune k/threshold bằng dev only;
- manual audit low-confidence;
- freeze model revisions và environment lock;
- draft phương pháp/kết quả paper.

### 23–27/08

- candidate A/B/C reproducibility rerun;
- package/validate dry-run;
- kiểm ZIP trên máy sạch;
- contract/format submission public có kế hoạch;
- chọn candidate theo metric, CI, runtime và risk.

### 28–31/08

- public final theo lịch dashboard thực tế;
- không đổi parser/model sau freeze nếu không có regression test;
- archive submission ZIP, SHA-256, dashboard score/screenshot;
- hoàn thiện error analysis/paper.

### 01–03/09 private

- chỉ nộp cấu hình đã freeze;
- không probe table-ref/hyperparameter;
- mỗi lượt cần checklist + người duyệt;
- giữ ít nhất một lượt dự phòng nếu luật dashboard cho phép.

## 4. Experiment matrix tối thiểu

| Run | Sparse | Dense | Rerank | Program | Mục đích |
|---|---|---|---|---|---|
| R00 | BM25 | — | — | gold/manual | lower baseline |
| R01 | BM25 | BGE-M3 | — | gold/manual | giá trị dense/RRF |
| R02 | BM25 | BGE-M3 | bge-reranker | gold/manual | giá trị rerank |
| R03 | R02 + multi-view | BGE-M3 | bge | gold/manual | representation |
| R04 | BM25 | — | finance-card | gold/manual | constraint rerank |
| R05 | BM25 | SPLADE hoặc late | — | gold/manual | sparse/late value |
| R06 | BM25 cascade | late | finance-card | gold/manual | compute-aware candidate |
| E00 | gold table | — | — | scalar IR | program upper bound easy |
| E01 | gold table | — | — | raw pandas LLM | compare IR |
| E02 | R02 retrieved | — | — | typed IR | end-to-end |
| E03 | R03 retrieved | — | — | typed IR | final candidate |
| E04 | gold table | — | — | cohort/temporal/formula IR | hard grammar coverage |
| V00 | E04 | — | — | + cell/unit/formula verifier | verifier contribution |
| V01 | V00 | — | — | + cross-path/abstention | risk–coverage |
| S00 | frozen retrieval | — | — | QLoRA program DSL | synthetic training, sau P7 |

Mỗi run lưu `config.yaml`, `environment.json`, `per_question.jsonl`, `summary.json`, `errors.jsonl`,
runtime/VRAM và commit/data/model hash.

## 5. Quality checklist liên tục

Sau mỗi thay đổi code:

```powershell
.venv\Scripts\ruff.exe check src scripts tests
.venv\Scripts\mypy.exe src scripts
.venv\Scripts\python.exe -m pytest -q
```

Trước full corpus:

- smoke 100 bảng;
- sequential/parallel equality;
- không overwrite artefact đã freeze;
- disk/RAM estimate;
- checkpoint/resume strategy.

Trước submission:

- 1.012/1.012 complete;
- no errors/unresolved unit;
- every answer re-executed;
- table/doc/evidence set minimal và consistent;
- one JSON at ZIP root, all paths under data/;
- offline validator xanh trên máy sạch;
- model compliance evidence complete;
- active table-ref contract verified;
- ZIP SHA-256 và run ID archived.

## 6. Stop conditions

Dừng và không promote nếu:

- metric tăng chỉ trên test/public probing, không trên dev/manual set;
- parser thay đổi làm table identity/hash drift không giải thích;
- model vi phạm parameter/cutoff/license;
- unit source/target mơ hồ;
- output query không tái lập;
- full run có missing ID hoặc silent fallback;
- dense/rerank tăng retrieval proxy nhưng giảm downstream execution/answer;
- private submission chưa có checklist/approval.

## 7. Lệnh runbook chính

```powershell
# Local CPU
.venv\Scripts\python.exe scripts\00_audit_dataset.py
.venv\Scripts\python.exe scripts\10_build_manifest.py --workers 6
.venv\Scripts\python.exe scripts\20_build_bm25.py
.venv\Scripts\python.exe scripts\30_retrieve_questions.py --candidate-k 2000
.venv\Scripts\python.exe scripts\32_validate_retrieval.py
.venv\Scripts\python.exe scripts\60_sample_qrels.py --size 100
.venv\Scripts\python.exe scripts\61_build_qrels_pool.py --run bm25=outputs\retrieval.jsonl

# Inspect/export evidence
.venv\Scripts\python.exe scripts\31_export_evidence.py "DOC|table_1"

# GPU/Linux: xem notebook Kaggle
.venv\Scripts\python.exe scripts\22_build_dense.py --device cuda
.venv\Scripts\python.exe scripts\50_generate_programs.py

# Final offline gates
.venv\Scripts\python.exe scripts\40_validate_submission.py outputs/generation/submission.json
.venv\Scripts\python.exe scripts\41_package_submission.py outputs/generation/submission.json submissions/submission.zip
```

Không chạy hai lệnh cuối nếu generation chưa đủ 1.012 hoặc table-ref contract chưa được xác nhận.
