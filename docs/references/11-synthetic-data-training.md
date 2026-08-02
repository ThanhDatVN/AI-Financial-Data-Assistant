# Synthetic supervision, hard negatives và training

## 1. Gate trước khi sinh dữ liệu

Synthetic data chỉ có giá trị sau khi:

- split entity/year/template/table fingerprint đã khóa;
- parser/evidence coordinates ổn định;
- typed program thực thi deterministic;
- một human-gold calibration set tồn tại;
- license và rule cho phép dùng corpus/model tương ứng.

Nếu sinh trước split, cùng template/table có thể rò từ train sang dev/test. Nếu không có gold nhỏ,
không đo được generator đang khuếch đại lỗi nào.

## 2. Pipeline đề xuất

```text
train-only facts
  → sample scenario/formula/coverage hole
  → generate typed program first
  → execute to exact answer + intermediate facts
  → render Vietnamese question
  → reverse parse question back to constraints
  → verify evidence/program/answer round trip
  → adversarial near-duplicate check
  → deduplicate/fingerprint
  → human audit sample
```

Program-first tránh LLM bịa answer rồi tìm công thức hợp thức hóa. Hướng executable templates và
step alignment được hỗ trợ bởi [FinChain](https://aclanthology.org/2026.acl-long.662/); synthetic
financial QA end-to-end cũng được nghiên cứu tại
[EACL Industry 2026](https://aclanthology.org/2026.eacl-industry.51/).

## 3. Loại dữ liệu cần sinh

| Loại | Mục tiêu | Validation |
|---|---|---|
| positive QA/program | phủ operator/scenario hiếm | execute + reverse constraints |
| retrieval positive view | align query với partial table | same table/cell sufficient |
| hard negative | near-duplicate khác entity/year/scope/metric | exactly one controlled mismatch |
| counterfactual unanswerable | train abstention | exhaustive corpus check |
| OCR perturbation | robustness accent/spacing/separator | answer invariant nếu semantics giữ |
| metamorphic pair | test algebra/scale/order | known relation between outputs |
| minimal pair | đổi một entity/year/operator | evidence/answer thay đúng dependency |

[QGpT](https://arxiv.org/abs/2508.06168) gợi ý sinh câu hỏi từ partial table để cải thiện retrieval.
Áp dụng chỉ cho train; generated question không được chứa source value nếu production prompt cũng cấm.

## 4. Hard-negative curriculum

Tạo negative theo độ gần:

1. random khác domain;
2. cùng metric khác company;
3. cùng company/metric khác year;
4. cùng company/year khác scope;
5. cùng label nhưng subtotal/definition khác;
6. bảng đúng một phần nhưng thiếu một operand;
7. conflict/restated value.

Không dùng LLM tạo negative tự do rồi mặc định đúng. Mỗi negative phải có machine-checkable violation
card. Mix random + hard; theo dõi source shortcut và topic drift. Chỉ promote khi retrieval dev và
critical slices cùng tăng.

## 5. Quality filters

- schema/grammar parse 100%;
- exact execution, no NaN/Inf/zero-divide;
- all operands grounded và route-able;
- formula dimension hợp lệ;
- generated Vietnamese natural nhưng không thêm assumption;
- question → constraints round-trip exact;
- no near-duplicate crossing split;
- distribution cap để lookup/sum không lấn grammar hiếm;
- manual double-review một stratified sample mỗi generator version.

Reject rate là metric chất lượng, không phải mục tiêu cần hạ bằng cách nới validator.

## 6. Fine-tuning dưới giới hạn ≤14B

Thứ tự:

1. prompt/grammar baseline 7B;
2. supervised fine-tune/QLoRA trên program JSON train-only;
3. mix gold + synthetic với weight và provenance;
4. evaluate exact same frozen dev slices;
5. preference/repair training chỉ từ adjudicated execution traces;
6. calibration/abstention head hoặc lightweight calibrator tách biệt.

Không train model sinh raw Pandas làm production target. Target là typed DSL có schema nhỏ; compiler
và executor vẫn deterministic. Mọi checkpoint ghi base revision, adapter hash, training rows/hash,
seed, hyperparameters, license và hardware.

## 7. Khi không nên dùng synthetic data

- chưa có leakage-safe split/qrels;
- parser/cell identity đang đổi;
- formula definition chưa được review;
- improvement chỉ thấy trên synthetic dev;
- generator cần proprietary model bị rule cấm;
- compute budget làm giảm số ablation/evaluation quan trọng hơn.
