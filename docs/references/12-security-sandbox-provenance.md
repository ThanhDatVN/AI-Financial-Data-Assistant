# Security, sandbox, provenance và hosted execution

## 1. Threat model

Đầu vào không tin cậy gồm OCR/HTML, câu hỏi, model output, downloaded weights và notebook environment.
Các rủi ro chính:

- generated code đọc file khác, gọi network, fork process hoặc tiêu tốn CPU/RAM;
- prompt injection nằm trong report/table text;
- pickle/model artifact độc hại hoặc revision drift;
- path traversal/ZIP packaging;
- data exfiltration qua hosted notebook/log;
- retrieval poisoning, conflicting/irrelevant evidence;
- silent fallback tạo answer không có trace.

AST allowlist không phải security boundary đầy đủ. Process timeout cũng không tự chặn filesystem,
network, child process hoặc kernel attack surface.

## 2. Kiến trúc ưu tiên

1. Model chỉ sinh typed JSON DSL.
2. Parser reject unknown field/operator.
3. Deterministic interpreter/compiler thao tác trên immutable evidence object.
4. Không expose Python object, filesystem path, network API hoặc dynamic import cho DSL.
5. Resource limit theo operation/input size trước khi chạy.
6. Trace mọi selected fact/operator/result.

Nếu DSL đủ biểu đạt 70 grammar local, production không có lý do chạy arbitrary Python. Raw Pandas chỉ
dùng trong controlled research ablation.

## 3. Nếu bắt buộc chạy code sinh tự do

| Lớp | Yêu cầu |
|---|---|
| process | non-root, no-new-privileges, PID/process limit |
| filesystem | ephemeral, read-only input, output dir quota, no host mounts |
| network | deny-all egress/ingress |
| resources | wall timeout, CPU quota, RAM limit, output-size limit |
| syscall/kernel | seccomp + container isolation; stronger runtime khi threat cao |
| artifacts | plain JSON/Parquet; không load untrusted pickle |
| audit | image digest, command, input/output hash, exit reason |

[gVisor security model](https://gvisor.dev/docs/architecture_guide/security/) giảm direct host-kernel
surface bằng user-space application kernel nhưng có overhead và vẫn là defense-in-depth.
[Firecracker](https://firecracker-microvm.github.io/) dùng KVM microVM với device model tối giản, phù
hợp workload multi-tenant/rủi ro cao hơn nhưng vận hành phức tạp. Cả hai là deployment option, không
phải yêu cầu cho local trusted test; trên Windows hiện tại executor chỉ là partial isolation.

## 4. Prompt/corpus isolation

- system/developer instruction không trộn với report text;
- report/table được delimit và luôn coi là data, không phải instruction;
- schema-only generation hạn chế source value và malicious strings;
- HTML được parse thành text/cells, không execute script/link;
- retrieved evidence không được thay config/tool permissions;
- conflict/poisoning signal hạ confidence, không được model tự chọn tùy ý.

[SafeRAG](https://arxiv.org/abs/2501.18636) là nguồn tham khảo cho noise/conflict/attack evaluation;
không ngoại suy score sang corpus này.

## 5. Model và dependency supply chain

- pin model repo + immutable revision SHA + license + parameter count;
- prefer `safetensors`; hash từng artifact quan trọng;
- pin Python/runtime package versions trong hosted run;
- không cài từ unknown fork hoặc execute remote notebook cell không review;
- xuất `environment.json`, GPU/runtime detection và model hash;
- scan secrets khỏi notebook/output trước archive/upload.

AWQ/GPTQ file không làm model “an toàn”; lượng tử hóa chỉ là representation/inference trade-off.

## 6. Hosted Kaggle/Colab

- upload chỉ corpus/artifact được phép theo license và rule;
- secrets qua secret store, không hard-code;
- checkpoint sau index, generation batches và evaluation;
- verify checksum sau upload/download;
- runtime/GPU/quota được detect, không hard-code hoặc hứa trước;
- notebook fail closed nếu model >14B, revision/license thiếu, manifest hash lệch hoặc output incomplete;
- không tự động submit dashboard hay dùng private quota.

## 7. Gate hiện tại

| Hạng mục | Hiện trạng | Claim hợp lệ |
|---|---|---|
| AST guard | có | chặn grammar ngoài allowlist |
| separate spawn + timeout | có | cô lập process cơ bản/kill timeout |
| POSIX memory limit | optional | không áp dụng tương đương Windows |
| network deny/read-only FS/CPU quota | chưa có | không gọi adversarial sandbox |
| typed DSL/compiler | có baseline | production path ưu tiên |
| immutable model/dependency lock | notebook cần chạy | `READY-NOT-RUN` |

Promotion production chỉ khi threat model và deployment boundary được review; test pass của AST parser
không thay penetration/isolation test.
