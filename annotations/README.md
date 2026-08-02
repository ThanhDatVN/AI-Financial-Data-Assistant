# Internal annotations

`qrels_template.jsonl` là mẫu schema v2 gồm 100 câu **chưa gán nhãn**, sinh deterministic bởi
`scripts/60_sample_qrels.py` với seed 20260802. SHA-256 hiện tại:
`1584fe25a4ba7889d822f93ecaa8916e00fe57f01700172f09ffbda7136a0114`.

`qrels_pool.jsonl` là pool khởi tạo từ frozen BM25 retrieval v2, gồm 100 câu/1.998 candidate, SHA-256
`8d3519bec6dcc6baf08ba4b1d01f8e445e72c0140c2f59140fe2544049ac4eba`. Đây **chưa phải diverse
pool**: phải chạy lại `scripts/61_build_qrels_pool.py` với dense/late/rerank runs trước annotation cuối.

Không dùng hai file này để báo metric cho tới khi các dòng qrels đạt `status=adjudicated` theo
[hướng dẫn](../docs/08-huong-dan-gan-nhan-qrels.md). Pool dùng `relevance/sufficiency`; gold cuối nằm
trong template v2 sau adjudication.

Khi freeze một phiên bản đã adjudicate, đổi tên kèm version, ghi SHA-256 vào nhật ký và không sửa tại
chỗ sau khi bắt đầu ablation.
