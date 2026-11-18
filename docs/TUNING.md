# Chọn tham số trên validation

Mọi lựa chọn của tracker dùng tập val: nửa sau của từng sequence train (71 sequence, 69.612 frame), cùng một detection cache YOLOX-X và mask glare. Cấu hình được khóa rồi đánh giá một lần trên test.

## Nguồn của từng giá trị

Các lượt SMAC3 trước đó trên val cho ra giá trị lẻ cho các ngưỡng overlap, vùng đầu và giới hạn glare. Cấu hình cuối làm tròn các giá trị này để giảm số chữ số được fit.

| Tham số | Giá trị | Nguồn |
|---|---:|---|
| `DET` | 0,10 | Score thấp nhất trong cache được dùng |
| `INIT` | 0,60 | Lưới trên val, 0,50–0,70 |
| `NMS` | 0,60 | SMAC3 cho 0,602, làm tròn |
| `IOU` | 0,30 | SMAC3 cho 0,267, làm tròn |
| `MIN_HITS` | 3 | SMAC3 |
| `MAX_AGE` | 120 frame | Cấu hình khởi đầu của lượt tìm kiếm |
| `HEAD_FRACTION` | 0,25 | SMAC3 cho 0,260, làm tròn |
| `EV` | 0,90 | SMAC3 cho 0,872, làm tròn |
| `GLARE_COAST_SECONDS` | 1 s | SMAC3 cho 1,16 s, làm tròn |

`INIT` là giá trị duy nhất chọn bằng lưới trong thiết kế cuối. Với các luật cuối, NMS 0,50, 0,55 và 0,65, giới hạn glare 2 và 4 giây, và `MIN_HITS` = 2 đều cho MOTA val thấp hơn giá trị đã chọn.

## Tiêu chí chọn

Chọn theo **MOTA trên val**. Trong các cấu hình chỉ xuất box dự đoán khi có glare, cấu hình cuối có MOTA val cao nhất: MOTA 80,29, HOTA 57,98, IDF1 77,13, 58 ID switch.

## Lệnh `tune`

`python -m exsort tune` chạy lại quy trình tìm kiếm trên val cho tracker cuối:

- SMAC3 2.4.1, surrogate random forest, acquisition Log Expected Improvement. Đề xuất tuần tự; mỗi trial replay **toàn bộ** val trước khi có đề xuất tiếp theo.
- Điểm khởi đầu là `--initial-config` (mặc định `configs/final.json`). Miền tìm kiếm, số trial và seed nằm trong `--search` (mặc định [configs/search.json](../configs/search.json)). Tham số không có trong miền tìm kiếm giữ giá trị của cấu hình khởi đầu.
- Cost của SMAC là `100 − MOTA`. Xếp hạng cuối: MOTA cao nhất; hòa thì HOTA, rồi IDF1, rồi ít ID switch hơn.
- Lệnh từ chối manifest không phải val. Kết quả từng trial, `ranked_results.json`, `best.json` và `selection.json` đã khóa được lưu trong thư mục output.

`configs/search.json` là miền mặc định quanh cấu hình cuối, dùng khi muốn tìm lại trên dữ liệu khác. Đây không phải bản ghi của các lượt tìm kiếm cũ. Đổi miền tìm kiếm hoặc số trial thì dùng thư mục output mới. SMAC chạy với ngân sách hữu hạn nên không bảo đảm tối ưu toàn cục.

## Giới hạn của miền tìm kiếm

Cache đã áp NMS 0,65, giữ tối đa 100 detection mỗi frame và bỏ score dưới 0,001. Vì vậy cần `NMS ≤ 0,65` và `DET ≥ 0,001`. Muốn vượt các giới hạn này phải tạo cache mới.
