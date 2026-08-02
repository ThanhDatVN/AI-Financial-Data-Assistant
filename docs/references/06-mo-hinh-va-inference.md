# 06 — Mô hình, inference và kiểm soát tuân thủ

## 1. Hàng rào bắt buộc

Theo đặc tả local, mọi PLM/LLM phải công khai model và/hoặc dữ liệu huấn luyện, phát hành trước
01/06/2026 (UTC+07), có tổng kích thước **≤14B**; model đóng bị cấm. Vì vậy:

- không gọi OpenAI/Gemini/Claude hoặc endpoint proprietary ở bất kỳ stage nào;
- loại Qwen2.5 14B/Coder-14B vì model card ghi lần lượt khoảng 14,8B/14,7B, vượt ngưỡng cứng;
- không dựa vào tên marketing “14B” hoặc số active parameters của MoE;
- chưa pin revision SHA thì artefact chỉ là thử nghiệm, **không đủ điều kiện nộp**.

## 2. Baseline đã chọn

| Vai trò | Model | Fact từ model card | Trạng thái |
|---|---|---|---|
| NLU dự phòng | [`Qwen/Qwen2.5-7B-Instruct-AWQ`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ) | base 7,61B; AWQ; Apache-2.0 | hợp ngưỡng; chưa pin SHA |
| Sinh chương trình | [`Qwen/Qwen2.5-Coder-7B-Instruct-AWQ`](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-AWQ) | base 7,61B; AWQ; Apache-2.0 | hợp ngưỡng; chưa pin SHA |
| Dense | [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) | multilingual, 8.192 token, dense/sparse/multi-vector | ablation GPU |
| Reranker | [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3) | multilingual cross-encoder | ablation GPU |

Model 7B không được mặc định là vừa mọi GPU: AWQ, KV cache, context, CUDA kernel và tensor parallel
đều ảnh hưởng VRAM. Notebook phải smoke 5 câu trước khi chạy full.

## 3. Thiết kế inference

Pipeline mặc định dùng OpenAI-compatible endpoint cục bộ do vLLM phục vụ, nhưng **chỉ endpoint open
model đã đăng ký**. Script gửi `response_format.type=json_schema`; đây là giao diện structured output
hiện hành của [vLLM](https://docs.vllm.ai/en/stable/examples/features/structured_outputs/), thay cho
`GuidedDecodingParams` cũ.

Schema hiện ép model sinh typed IR, không nhận raw Pandas. IR được parse/ground theo evidence rồi compiler
tạo query; compatibility của schema đệ quy với model/backend cụ thể vẫn phải qua smoke GPU.

LLM nhận question, target unit, coordinate/header/row label; không nhận chữ số nguồn. LLM có thể đọc
số trong chính câu hỏi (năm, ngưỡng, số lượng yêu cầu) nhưng không được chép giá trị ô. Giá trị được
đọc từ CSV và tính bằng executor. Temperature chính bằng 0; seed, prompt hash và schema hash phải lưu.

vLLM chạy trên Linux/CUDA; không coi Windows native là môi trường production. File
`requirements-gpu.txt` pin `vllm==0.25.1`, phiên bản được đối chiếu trên PyPI ngày 02/08/2026. Đây là
pin của dự án tại thời điểm rà soát, không phải tuyên bố tương thích với mọi hosted image.

## 4. Kaggle/Colab

[Colab FAQ](https://research.google.com/colaboratory/faq.html) nói rõ loại GPU, quota, timeout và mức
khả dụng thay đổi; Kaggle cũng cần kiểm policy/UI tại thời điểm chạy. Do đó không ghi “2×T4”, “P100”,
“30 giờ/tuần” hoặc throughput ước lượng như một cam kết.

Quy trình an toàn:

1. in `nvidia-smi`, RAM, disk, Python/torch/CUDA và `pip freeze`;
2. tải model theo revision đã duyệt hoặc attach checkpoint;
3. build dense/reranker, giải phóng VRAM;
4. khởi động vLLM với TP=1 trước, context 4.096–8.192 tùy smoke;
5. sinh 5 câu, kiểm JSON/schema/execution/error log;
6. chạy checkpoint theo question ID, copy artefact ra persistent storage;
7. validate 1.012 câu và ZIP clean-room trước upload.

Notebook đầy đủ và hướng dẫn nằm tại [notebooks/README.md](../../notebooks/README.md) và
[docs/04-huong-dan-kaggle-colab.md](../04-huong-dan-kaggle-colab.md).

## 5. Model register — gate trước submission

| Vai trò | Base/repo thực dùng | Revision SHA | Params | License | Release < cutoff | Env/log | Duyệt |
|---|---|---|---:|---|---|---|---|
| NLU | Qwen2.5-7B-Instruct / AWQ | `TODO` | 7,61B | Apache-2.0 | có | `TODO` | chặn |
| IR/code | Qwen2.5-Coder-7B-Instruct / AWQ | `TODO` | 7,61B | Apache-2.0 | có | `TODO` | chặn |
| Dense | BAAI/bge-m3 | `TODO` | 568M theo config/card | MIT | có | `TODO` | chặn nếu dùng |
| Reranker | BAAI/bge-reranker-v2-m3 | `TODO` | 568M theo config/card | Apache-2.0 | có | `TODO` | chặn nếu dùng |

Không tự điền SHA `main`: tại runtime phải resolve commit thực, lưu cùng output và cập nhật bảng. Nếu
không dùng một model trong run cuối, xóa model đó khỏi registry/working paper thay vì kê khai thừa.

## 6. Failure policy

- JSON sai schema: ghi error, không sửa bằng model đóng.
- OOM: giảm context/batch/TP theo log, không đổi sang model >14B.
- Thiếu revision/license: dừng trước submission.
- Runtime reset: resume từ checkpoint; không trộn completion của model/config khác trong cùng run.
- Executor lỗi hoặc answer không hữu hạn: fail closed, không điền 0/NaN làm mặc định.
- Table-ref/tolerance chưa xác nhận: giữ cấu hình companion để dev parity nhưng chặn upload chính thức.

## 7. Việc chưa có bằng chứng local

- chất lượng dense/reranker trên ViFinQA;
- chất lượng/throughput Qwen AWQ trên GPU được cấp;
- sai khác AWQ so với BF16/FP16;
- khả năng đáp ứng full operator cho aggregate/rank/panel;
- table-ref grammar và tolerance của dashboard.

Các mục trên là `pending external evidence`, không được ghi là đã đạt trong báo cáo.
