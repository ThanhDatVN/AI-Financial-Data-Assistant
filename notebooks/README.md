# Notebook GPU/hosted runtime

Hai notebook là lớp điều phối mỏng quanh các script đã được test trong repository. Không sửa trực
tiếp cell để đổi logic; hãy sửa module/script, chạy test, rồi chạy lại notebook.

- `01_colab_prepare_and_index.ipynb`: tải corpus, audit, build manifest/BM25 và lưu artefact vào
  Google Drive. CPU runtime đủ dùng; GPU không bắt buộc.
- `02_kaggle_dense_and_generate.ipynb`: build BGE-M3/FAISS, hybrid retrieval, khởi động
  Qwen2.5-Coder-7B-Instruct-AWQ bằng vLLM, sinh chương trình có checkpoint, validate và đóng ZIP.

Trước khi chạy, đọc `docs/04-huong-dan-kaggle-colab.md`. Hạn mức và loại GPU của Colab/Kaggle
thay đổi theo tài khoản và thời điểm; notebook tự phát hiện phần cứng và không giả định T4/P100.
