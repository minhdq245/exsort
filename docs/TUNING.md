# Tuning protocol và lịch sử nghiên cứu

## Mục tiêu

Tối đa **pooled HOTA trên full tracker val**, với MOTA không thấp hơn baseline đã khai báo. Chỉ số chất lượng tính theo %; count giữ số nguyên. Nếu hai ứng viên hòa HOTA, ưu tiên IDF1, MOTA, rồi ít IDSW.

HOTA pooled do TrackEval `combine_sequences` tạo; không lấy trung bình HOTA từng sequence. Cũng cần báo sequence mean và theo từng cảnh vì morongmo chiếm 36.726/69.612≈52,76% frame val. Temporal val nằm trong cùng scene với detector train; nó không thay thế đánh giá generalization sang scene mới.

## Cách Bayesian optimization hoạt động ở đây

SMAC3 fit random forest từ các cặp (tham số, validation cost). Acquisition Log Expected Improvement cân bằng khảo sát nơi chưa chắc chắn và khai thác vùng có điểm hứa hẹn. Các đề xuất đầu gồm điểm khởi đầu và thiết kế Sobol. Mỗi lần replay full val cung cấp một quan sát mới. Không có gradient qua tracker; detector không được train lại trong quá trình này.

CLI public dùng `ask` → ghi `pending.json` → replay/cache kết quả → `tell` → lưu runhistory. Nếu bị ngắt, lệnh tương tự với cùng output tiếp tục từ pending trial và các sequence/block đã lưu. Per-sequence replay kết quả được khóa bởi cache block SHA, cấu hình, source và GT signature. Không dùng lại score cũ sau thay GT.

Cost CLI:

```text
eligible:    100 - HOTA
ineligible:  200 + MOTA_floor - MOTA
```

Đây là scalar penalty cụ thể cho constrained selection, không phải chứng minh tối ưu Pareto hoặc constrained Bayesian optimization xác suất. Xếp hạng cuối dùng trực tiếp rule HOTA/MOTA; baseline luôn còn trong tập ứng viên. MOTA floor đo lại trên val hiện hành.

CLI cố định cấu trúc trong numeric tuning, đề xuất tuần tự để mỗi trial nhận feedback từ trial trước. `--workers` chỉ song song các sequence. Cách này dễ audit và tránh phải đọc lại ảnh/GPU. Không khẳng định mọi trial luôn cải thiện hoặc số trial nhỏ luôn tìm được optimum.

## Lịch sử trước bản public

Các bước nghiên cứu đã thực hiện với detector YOLOX-X epoch15, full dense val 71 sequence, trước sửa GT cuối tháng 10/đầu tháng 11:

1. Vòng ban đầu: tắt lamp update, thêm low-confidence continuation, giữ các setting còn lại; SMAC 64 full-val trials, RF surrogate, có điểm động cơ chế + Sobol warm start; batch hai proposal.
2. Vòng cải tiến lifecycle/birth/coast: thử tách tentative/confirmed, thay chính sách birth và giới hạn output lúc mất dấu. Phát hiện phiên bản confidence-dependent R ban đầu bị ORU thay thế; loại kết quả nhánh lỗi, sửa bằng truyền explicit R và tạo lineage mới. Bản chọn cuối không dùng nhánh R này.
3. Vòng localization: ablation 34 cấu hình về raw/posterior/blend, low association/output, HMIoU/score fusion, IoM birth, INIT và area/aspect filter. Giữ tối đa bốn cấu trúc có tính đa dạng theo HOTA/MOTA/IDF1, rồi SMAC local 24 trial, seed `20261030`, batch ba proposal. Tiếp theo kiểm tra lân cận tọa độ trên val.
4. Vòng recovery: thử cặp unique lost-ID recovery và bỏ lost-track birth veto trên các anchor cấu trúc; so sánh từng thành phần và tương tác. Thử 12 cấu hình recovery và sáu lân cận DET±0,01, IOU±0,01, MAX_AGE±20, cắt theo miền cho phép. Không thêm khoảng cách hình học mới: dùng lại axis gate.
5. Chọn HOTA cao nhất trong các ứng viên đạt MOTA baseline trước vòng; đóng băng source/config, sau đó chạy một final configuration trên test của vòng đó.

Kết quả cuối có **84 cấu hình duy nhất** trong lineage vòng cuối, bao gồm cache kế thừa; đây không phải 84 trial SMAC mới. Số ứng viên các giai đoạn không cộng thẳng được do trùng cấu hình, baseline và score reuse. `history/validation_candidates_20261029.json` giữ tham số/score của từng ứng viên. Không dùng các score đó để lựa chọn trên GT mới.

CLI public tái hiện **quy trình làm thực nghiệm** bằng các lệnh portable, không giả vờ rằng một lệnh numeric tune chạy lại toàn bộ lịch sử chọn cấu trúc. Để làm nghiên cứu mới: khai báo trước ablations và ngân sách; chạy ablation val, chọn cấu trúc, tune số, kiểm tra local nếu đã định trước; giữ toàn bộ ứng viên đã thử, rồi khóa một selection trước test.

## Chứng cứ tương tác và giới hạn

Ở một anchor val cũ: recovery+defer đạt HOTA≈56,154; bỏ recovery chỉ defer≈54,098; recovery nhưng vẫn veto≈55,108. Điều này gợi ý hai cơ chế tương tác ở dữ liệu này, không chứng minh tính phổ quát.

Val trước vòng cuối: HOTA54,9033, IDF1 71,9428, MOTA76,8377, IDSW98, FP11.652, FN31.651.

Val bản chọn: HOTA56,1608, IDF1 73,9375, MOTA77,5454, IDSW75, FP10.980, FN31.020.

Paired sequence bootstrap 500 lần cho delta HOTA có percentiles 2,5/50/97,5%: `[-0.00695, 1.22865, 1.51215]` điểm %. Đây là bootstrap mô tả sau adaptive selection; sequence cùng scene tương quan, cận dưới hơi âm. Không dùng nó để tuyên bố một kiểm định xác nhận độc lập hoặc thắng tuyệt đối.

Test đã có lịch sử tiếp xúc trong các vòng trước. “Final config chọn bằng val trước test của vòng hiện tại” không biến test cũ thành chưa từng được xem. GT của 23 sequence tiếp tục được sửa 31/10–01/11; số liệu 28–29/10 phải gắn snapshot cũ. Muốn so sánh hiện tại, chấm lại mọi method bằng cùng GT và cùng detector protocol.

## Báo cáo cần lưu

- Split, temporal rule, số sequence/frame/GT thực tế và SHA phiên bản GT.
- Detector SHA, input H×W, preprocessing, FP16/FP32, NMS/cap/score floor.
- Mask V/S/resize; mọi tham số và switch cấu trúc, Kalman dependency.
- Optimizer/version/seed/budget, search bounds, baseline floor, tie-break và mọi vòng thử.
- Pooled HOTA/DetA/AssA/IDF1/MOTA, **IDSW**, FP/FN, các metric phụ của TrackEval và per-sequence results.
- Tổng thời gian cache riêng với thời gian tracking/scoring; throughput đo thực, cấu hình phần cứng, peak VRAM.
- Test exposure, thay GT và khác biệt preprocessing giữa các baseline.

Nguồn chính thức: [SMAC3](https://github.com/automl/SMAC3), [TrackEval](https://github.com/JonathonLuiten/TrackEval), [TrackTrack](https://github.com/kamkyu94/TrackTrack), [OC-SORT](https://github.com/noahcao/OC_SORT).
