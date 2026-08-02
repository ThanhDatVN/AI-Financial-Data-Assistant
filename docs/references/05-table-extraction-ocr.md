# 05 — Parser bảng OCR đã cấu trúc và provenance

## 1. Phạm vi thật

Release cung cấp `.txt` với mốc `===== PAGE n =====` và HTML `<table>`. Pipeline không có ảnh/PDF để
đánh giá OCR lại, vì vậy vision OCR/VLM bị loại khỏi critical path. Nhiệm vụ là segment, parse
rowspan/colspan, nhận diện header/unit, normalize cell và giữ coordinate quay về raw source.

## 2. Audit toàn corpus đã chạy

| Kiểm tra | Kết quả release local |
|---|---:|
| báo cáo | 1.973 |
| raw inline tables | 146.246 |
| opening/closing/matched tag mismatch | 0 |
| table refs trong full manifest | 146.246 unique |
| parsed empty/malformed | 0 |
| báo cáo không có table | 8 (đều thư giải trình PRT) |
| trang median / p95 / max | 57 / 97 / 206 |
| bảng mỗi report median / p95 / max | 70 / 116 / 248 |

Con số 143.815 ở dataset card/companion `docs/data.md` không khớp raw release. README companion hiện
ghi 146.246. Dùng audit local + checksum làm ground truth kỹ thuật cho bản đã tải.

## 3. Provenance tối thiểu

Mỗi record giữ:

- `doc_id`, ticker, report year, scope;
- raw ordinal 1-based và `table_ref` configurable;
- page number, char offset/context;
- raw table hash và source file path;
- header/row labels, unit hint và parse warnings;
- `dashboard_verified=false` cho tới khi contract được test.

Không suy diễn `|350` trong ví dụ đặc tả là ordinal khi chưa có xác nhận. Companion dùng
`doc_name|table_N` và fixture 1-based; dự án dùng format đó cho compatibility, nhưng giữ metadata đủ để
đổi adapter nếu dashboard dùng char/line position.

## 4. Parser và normalization

| Hiện tượng | Xử lý |
|---|---|
| inline/multiline HTML | segment cân bằng, parse bằng `lxml` |
| rowspan/colspan | expand thành ma trận chữ nhật, giữ raw coordinate |
| unit ngoài bảng | tìm context gần bảng; không thấy thì `UNKNOWN` |
| unit trong header | extract hint nhưng không xóa raw header |
| `.` phân cách nghìn | locale-aware parser, không `float()` trực tiếp |
| `(1.234)` | số âm |
| dash | zero có flag riêng, không đánh đồng missing |
| text/OCR không chắc là số | từ chối parse thay vì đoán |
| header nhiều tầng | giữ nhiều header row và flattened label có truy vết |

Full manifest strict có 29.068/146.246 bảng `UNKNOWN` unit sau khi chặn context/unit rò từ bảng
liền trước. Đây là table-default; evidence còn resolve override theo column/row label. Không gán VND
mặc định chỉ để tăng coverage.

## 5. Evidence schema

CSV long-form cố định tách identity khỏi số:

```text
row_index, column_index, row_label, column_label,
raw_value, numeric_value, source_unit, base_value
```

`base_value` là giá trị sau scale nguồn; target conversion diễn ra trong program/compiler. Prompt LLM
chỉ nhận coordinate/labels, không nhận `raw_value`, `numeric_value` hay `base_value`.

## 6. Quality gates

- opening = closing = matched table count trên toàn corpus;
- table-ref và source hash duy nhất/ổn định;
- parser không nuốt exception; warning có doc/table coordinate;
- round-trip fixture cho rowspan/colspan/header/unit;
- number parser test dấu nghìn, ngoặc âm, dash và refusal;
- real-corpus integration khóa một cell, unit conversion và answer;
- source file không đổi mà manifest hash đổi thì fail reproducibility;
- table-ref adapter phải có một dashboard probe trước submission.

## 7. Phần còn thiếu

USD/million USD và mixed-unit row/column đã có regression. Unit anchor sau bảng, block unit nằm trong
body, EUR/ngoại tệ khác, continuation-table merge và manifest shard/resume chưa có coverage đầy đủ.
Đây là regression backlog, không phải tính năng đã hoàn tất.
