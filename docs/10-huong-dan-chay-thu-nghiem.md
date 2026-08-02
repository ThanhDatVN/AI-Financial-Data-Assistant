# Hướng dẫn chạy thử nghiệm GPU từng bước

Tài liệu này áp dụng cho release ViFinQA hiện đã audit: 1.012 câu hỏi và 146.246 bảng. Kaggle GPU là
đường chạy chính. Không chạy full generation trước khi smoke trả đúng 5 prediction, 0 error và 5
program trace.

## 0. Trạng thái trước khi bắt đầu

Phần local đã pass, nhưng working tree hiện chưa được commit/push. Không dùng cell clone GitHub cho
final run trước khi đóng băng code, vì Kaggle sẽ lấy code cũ.

Không đưa `data/`, `outputs/`, model, token hoặc file `.env` vào Git. Chỉ stage code/tài liệu:

```powershell
git status --short
git add .gitignore README.md annotations configs docs notebooks pyproject.toml requirements.txt requirements.lock.txt requirements-gpu.txt scripts src tests cuoc-thi-text-to-pandas-bctc.md runs/20260802_RUN012_local_preflight/handoff_manifest.json
git diff --cached --stat
git diff --cached --check
git commit -m "Prepare reproducible ViFinQA GPU experiment"
git branch --show-current
git push -u origin <ten-branch-vua-in-ra>
git rev-parse HEAD
git status --short
```

Trước `commit`, phải đọc `git diff --cached`; sau `push`, ghi lại Git SHA. `git status --short` có thể
còn liệt kê artefact ignored theo cấu hình local, nhưng không được còn thay đổi code cần chạy.

## 1. Chuẩn bị hai Kaggle Dataset input

Tạo hai dataset private bằng Kaggle UI hoặc Kaggle API. Giữ đúng cấu trúc sau; tên slug khác thì chỉ
sửa hai path trong cell `kaggle-paths`.

```text
/kaggle/input/vifinqa/
└── ViFinQA/
    ├── code_stock.csv
    ├── questions/questions.jsonl
    └── ... toàn bộ raw corpus ...

/kaggle/input/vifinqa-artifacts/
└── data/
    ├── processed/
    │   ├── table_manifest.jsonl
    │   ├── table_manifest.parquet
    │   └── table_manifest.metadata.json
    └── index/bm25/
        ├── data.csc.index.npy
        ├── indices.csc.index.npy
        ├── indptr.csc.index.npy
        ├── params.index.json
        ├── records.jsonl
        └── vocab.index.json
```

Dung lượng local để đối chiếu:

| Input | Dung lượng xấp xỉ |
|---|---:|
| raw `ViFinQA` | 363 MiB |
| manifest JSONL + Parquet + metadata | 455 MiB |
| BM25 | 163 MiB |

Notebook tự hash toàn bộ ba manifest và sáu file BM25. Không bỏ qua assertion hash bằng cách sửa
notebook; mismatch nghĩa là upload thiếu, đổi tên, hoặc dùng artefact khác release.

## 2. Tạo Kaggle Notebook

1. Import `notebooks/02_kaggle_dense_and_generate.ipynb` từ Git commit vừa push.
2. Add Input: dataset `vifinqa` và `vifinqa-artifacts` ở trên.
3. Chọn GPU accelerator. Không giả định tên GPU; cell phần cứng sẽ in số GPU và VRAM.
4. Bật Internet cho experiment tải dependency/model. Nếu competition bắt buộc Internet off, phải
   chuẩn bị trước model snapshot và wheelhouse; không tự đổi sang API model đóng.
5. Không dùng `Run All` cho lần đầu. Chạy từng cell để dừng đúng gate.

Mọi output cần giữ nằm dưới `/kaggle/working`; input dưới `/kaggle/input` là read-only.

## 3. Chạy smoke theo từng cell

### Cell 1 — `kaggle-final-config`

Giữ nguyên lần đầu:

```python
os.environ["VIFINQA_FINAL_RUN"] = "0"
os.environ["VIFINQA_RUN_FULL"] = "0"
os.environ["VIFINQA_TP"] = "1"
```

Không sửa ba full commit SHA. Output phải là:

```text
final/full/tp: 0 0 1
```

### Cell 2 — `kaggle-hardware`

Điều kiện pass:

- `cuda True`;
- `gpus >= 1`;
- cell in đúng GPU và VRAM;
- không có assertion.

Nếu `cuda False`, vào Settings, bật GPU và restart session. Không chạy dense bằng CPU trên Kaggle.

### Cell 3 — `kaggle-install`

Cell clone/copy repo, cài project, BGE/FAISS/vLLM và ghi:

```text
/kaggle/working/runtime_environment.txt
```

Output `project revision` phải bằng Git SHA đã ghi ở bước 0. Với submission candidate, giá trị
`attached-archive-no-git-sha` là lỗi. Nếu pip yêu cầu restart kernel vì thay torch, restart rồi chạy
lại từ Cell 1; Cell 2 phải tiếp tục báo `cuda True`.

### Cell 4 — `kaggle-paths`

Cell kiểm path, SHA-256 và số câu hỏi. Output pass kết thúc bằng:

```text
inputs verified: ... questions= 1012
```

Nếu báo `Missing frozen artefact`, kiểm lại cây thư mục ở mục 1. Nếu báo `SHA-256 mismatch`, upload
lại đúng file local; không sửa expected hash.

### Cell 5 — `kaggle-dense`

Cell thực hiện ba việc:

1. build full BGE-M3/FAISS vào `/kaggle/working/artifacts/bge_m3`;
2. chạy hybrid BM25 + dense cho 1.012 câu;
3. chạy fail-closed retrieval QC.

Điều kiện pass:

- `config.json` có `tables = 146246`;
- `model_revision` bằng SHA đã khóa;
- `retrieval_hybrid_qc.json` có `passed=true`, `rows=1012`;
- output cuối in `hybrid retrieval verified` và retrieval SHA.

Ngay sau khi pass, lưu/version notebook output hoặc tải về tối thiểu:

```text
/kaggle/working/artifacts/bge_m3/
/kaggle/working/artifacts/retrieval_hybrid.jsonl
/kaggle/working/artifacts/retrieval_hybrid_qc.json
/kaggle/working/runtime_environment.txt
```

Đây là checkpoint đắt nhất; không đợi đến sau generation mới lưu.

### Cell 6 — `kaggle-vllm`

Cell khởi động Qwen AWQ tại `127.0.0.1:8000`. Pass khi in:

```text
vLLM ready
```

Nếu fail, đọc log:

```python
print(Path("/kaggle/working/vllm.log").read_text(encoding="utf-8")[-4000:])
```

Nếu OOM, ưu tiên theo thứ tự: giữ `TP=1`, giảm `--max-model-len` từ 8192 xuống 4096, rồi giảm
`--gpu-memory-utilization` từ 0.90 xuống 0.85. Ghi mọi thay đổi vào nhật ký run.

### Cell 7 — `kaggle-smoke`

Cell sinh và thực thi chương trình cho 5 câu đầu. Dòng tổng kết bắt buộc:

```text
smoke predictions/errors/traces: 5 0 5
```

Ngoài assertion tự động, kiểm thủ công ít nhất hai prediction được in:

- `answer` hữu hạn, không phải chuỗi lỗi/NaN;
- `relevant_tables` không rỗng và phù hợp ticker/năm;
- `pandas_query` chỉ dùng biến evidence đã chọn;
- CSV trong `generation/data/` tồn tại;
- đơn vị/tỷ lệ của câu hỏi khớp answer;
- `program_traces.jsonl` có đúng ID tương ứng.

Nếu có error, dừng. Đọc `generation/errors.jsonl`, sửa nguyên nhân và dùng output directory mới cho
smoke tiếp theo; không chạy full trên checkpoint đã chứa lỗi cũ.

## 4. Chuyển sang full run

Sau khi smoke pass, sửa duy nhất hai dòng trong Cell 1:

```python
os.environ["VIFINQA_RUN_FULL"] = "1"
os.environ["VIFINQA_FINAL_RUN"] = "0"  # full experiment, chưa phải submission candidate
```

Nếu đây là run đã được phép dùng làm submission candidate và rule/contract đã xác nhận, dùng:

```python
os.environ["VIFINQA_RUN_FULL"] = "1"
os.environ["VIFINQA_FINAL_RUN"] = "1"
```

Rerun **Cell 1 rồi chạy thẳng Cell 8**. Không rerun Cell 6 để tránh mở server thứ hai trên cùng port.
Cell 8 đọc lại hai cờ trực tiếp và resume 5 câu đã hoàn thành.

Pass cuối phải có:

```text
completed=1012/1012
valid submission: 1012 questions
/kaggle/working/submission.zip <bytes> <sha256>
```

Cell cũng yêu cầu `len(predictions)==1012` và `errors==0`. Packager từ chối ghi đè ZIP đã tồn tại;
nếu chạy một experiment mới, đổi output folder/ZIP name thay vì trộn checkpoint.

## 5. File phải tải xuống sau run

Tải hoặc Save Version với toàn bộ:

```text
/kaggle/working/submission.zip
/kaggle/working/runtime_environment.txt
/kaggle/working/vllm.log
/kaggle/working/artifacts/retrieval_hybrid.jsonl
/kaggle/working/artifacts/retrieval_hybrid_qc.json
/kaggle/working/artifacts/generation/run_metadata.json
/kaggle/working/artifacts/generation/predictions.jsonl
/kaggle/working/artifacts/generation/program_traces.jsonl
/kaggle/working/artifacts/generation/submission.json
/kaggle/working/artifacts/generation/errors.jsonl  # chỉ khi file tồn tại
```

Ghi vào `docs/nhat-ky-thi-nghiem.md`: Git SHA, GPU, package versions, model revisions, retrieval SHA,
ZIP SHA, thời gian từng stage, số success/error và mọi thay đổi do OOM.

## 6. Ma trận quyết định nhanh

| Hiện tượng | Hành động |
|---|---|
| Git SHA không đúng | dừng; push đúng commit rồi clone lại |
| input hash mismatch | dừng; upload lại artefact frozen |
| dense khác 146.246 bảng | dừng; không dùng index đó |
| retrieval QC fail | dừng; đọc failure counts, không start vLLM |
| vLLM không healthy | đọc log; xử lý VRAM/dependency trước smoke |
| smoke khác `5/0/5` | dừng; sửa và dùng generation directory mới |
| full thiếu ID/có error | không package/upload; resume hoặc tạo run mới |
| validator không báo 1.012 | không upload |
| table-ref/tolerance contract chưa xác nhận | chỉ lưu experiment; không submit dashboard |

Qrels hiện chưa được adjudicate, nên run GPU này đo coverage, failure rate, execution validity và tạo
artefact để gán nhãn; chưa được báo retrieval accuracy chính thức.
