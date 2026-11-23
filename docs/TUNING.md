# Chọn tham số trên validation

Mọi lựa chọn của tracker dùng tập val: nửa sau của từng sequence train (71 sequence, 69.612 frame), cùng một detection cache YOLOX-X và mask glare. Cấu hình được khóa rồi đánh giá một lần trên test.

## Lượt tìm kiếm

Một lượt SMAC3 chọn cùng lúc tám tham số: 100 trial, seed 20261123, mỗi trial replay toàn bộ 69.612 frame val. Miền tìm kiếm nằm trong [configs/search.json](../configs/search.json). `DET` cố định ở 0,10.

| Tham số | Giá trị được chọn | Miền |
|---|---:|---|
| `DET` | 0,10 | Cố định |
| `INIT` | 0,60 | 0,50–0,70 |
| `NMS` | 0,60 | 0,30–0,65 |
| `IOU` | 0,30 | 0,15–0,50 |
| `MIN_HITS` | 3 | 2–5 |
| `MAX_AGE` | 120 frame | 20–180 |
| `HEAD_FRACTION` | 0,25 | 0,15–0,40 |
| `EV` | 0,90 | 0,30–0,90 |
| `GLARE_COAST_SECONDS` | 1 s | 1–4 |

Tám cấu hình Sobol và một cấu hình khởi đầu (lấy từ các lần chạy sơ bộ trên cùng tập val) được chạy trước; SMAC3 đề xuất 91 cấu hình còn lại.

## Tiêu chí chọn

HOTA và MOTA là hai metric chính. Cost của một cấu hình là `200 − HOTA − MOTA` trên val; chọn cấu hình có cost thấp nhất, tức tổng HOTA + MOTA cao nhất trong 100 trial. Kết quả là cấu hình khởi đầu: HOTA 57,98, MOTA 80,29, IDF1 77,13, 58 ID switch. Cấu hình tốt nhất do SMAC3 đề xuất đạt HOTA 57,98, MOTA 80,26.

## Lệnh `tune`

`python -m exsort tune` chạy lại quy trình tìm kiếm trên val cho tracker cuối:

- SMAC3 2.4.1, surrogate random forest, acquisition Log Expected Improvement. Đề xuất tuần tự; mỗi trial replay **toàn bộ** val trước khi có đề xuất tiếp theo.
- Điểm khởi đầu là `--initial-config` (mặc định `configs/final.json`). Miền tìm kiếm, số trial và seed nằm trong `--search` (mặc định [configs/search.json](../configs/search.json)). Tham số không có trong miền tìm kiếm giữ giá trị của cấu hình khởi đầu.
- Cost của SMAC là `200 − HOTA − MOTA`. Xếp hạng cuối: HOTA + MOTA cao nhất (chưa làm tròn); hòa thì MOTA, rồi IDF1, rồi ít ID switch hơn.
- Lệnh từ chối manifest không phải val. Kết quả từng trial, `ranked_results.json`, `best.json` và `selection.json` đã khóa được lưu trong thư mục output.

Đổi miền tìm kiếm hoặc số trial thì dùng thư mục output mới. SMAC chạy với ngân sách hữu hạn nên không bảo đảm tối ưu toàn cục.

## Giới hạn của miền tìm kiếm

Cache dùng cho các kết quả đã công bố giữ detection có score từ 0,10. Lệnh `cache` trong repo mặc định giữ score từ 0,001, áp NMS 0,65 và giữ tối đa 100 detection mỗi frame. Vì vậy cần `NMS ≤ 0,65` và `DET ≥ 0,001`. Muốn vượt các giới hạn này phải tạo cache mới.
