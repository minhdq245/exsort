# Method: exsort được chọn

Đây là mô tả **configs/selected.json**, không phải mọi nhánh ablation còn được giữ trong code. `core.py` và `legacy.py` được lấy từ snapshot cuối của vòng nghiên cứu; chỉ đổi import thành package-relative. Hash gốc và nguồn dependency nằm trong `history/source_provenance.json`.

## Một frame được xử lý như thế nào?

1. **Tách detection theo confidence.** High có score≥DET; low có LOW≤score<DET. NMS riêng của tracker áp dụng lên high. Low không được sinh track mới.
2. **Dự đoán chuyển động.** OC-SORT Kalman dự đoán box; dùng state 7 chiều `[cx,cy,area,ratio,vx,vy,varea]` và cơ chế ORU của dependency vendored khi có lại quan sát sau mất dấu. Không chạy ReID/CMC trong exsort này.
3. **Ưu tiên track đã xác nhận.** Ghép high detection với box dự đoán bằng IoU/Hungarian; thử lại phần còn lại bằng box quan sát gần nhất. Ghép low detection với track confirmed còn thiếu bằng IoU. Sau đó high còn lại mới được ghép với tentative track. Ngưỡng IoU áp dụng sau assignment trong cấu hình được chọn (`ADMISSIBLE=false`); không gọi đây là assignment tối ưu có ràng buộc cardinality.
4. **Track tentative cần đủ high observations.** Sinh track với một high hit; chỉ confirmed khi đủ MIN_HITS. Tentative không nhận low và bị bỏ nếu bị mất một frame, nên ba high observations ở giai đoạn khởi tạo phải liên tiếp. Confirmed có thể được low detection tiếp tục duy trì.
5. **Giải thích mất detection bằng glare đo được.** Với track confirmed chưa ghép được, tìm pixel bão hòa trong dải trên cùng của box dự đoán, cao bằng HEAD_FRACTION×chiều cao box. Chỉ cần có pixel thỏa mask là `now=true`. Đây là tín hiệu hỗ trợ mất detection, không chứng minh chắc chắn có người. `lamp=false`: không kéo tâm box về tâm đèn.
6. **Tuổi ID và quyền xuất box tách nhau.** Mất detection không được glare giải thích sẽ tăng `unexplained`; quan sát thật reset về 0; glare giải thích làm counter không tăng trong frame đó. Track sống khi `unexplained≤MAX_AGE`. Quyền tiếp tục xuất box dự đoán chặt hơn: glare phải giải thích ở frame hiện tại, IoU(prediction,last real observation)≥EV, và số frame từ detection thật cuối≤MAX_AGE (`COAST_MODE=budget`). Counter có tên `unexplained` trong code tích lũy các frame mất chưa giải thích kể từ quan sát thật cuối; không reset chỉ vì một frame có glare.
7. **Sinh track và chống trùng.** Detection high chưa ghép chỉ được birth nếu score≥max(DET,INIT). Track còn quan sát thật có thể veto birth bằng cùng axis gate. Với `LOST_BIRTH=defer`, track đang mất dấu không veto, để người được nhìn thấy lại có cơ hội hình thành candidate.
8. **Khôi phục ID duy nhất sau xác nhận.** Candidate vừa đủ high hits, chưa từng confirmed trước đó, được so với các old confirmed/lost track còn sống. Nếu đúng **một** old track thỏa axis gate, candidate lấy ID cũ; old instance bị xóa trước output. Không có ứng viên hoặc có ≥2 thì giữ ID mới. `RECOVERY=true` là nhánh khôi phục mới; cờ `inherit=false` tắt cơ chế tự gọi ở legacy path, không có nghĩa final method không khôi phục ID.
9. **Xuất box.** Khi có observation thật, dùng Kalman **posterior**, không dùng trực tiếp box detection. Khi thiếu observation, chỉ xuất prediction nếu qua ba điều kiện ở bước 6. Không có TrackTrack area/aspect filter; không nội suy offline. Không xuất các frame trước thời điểm track được xác nhận.

## Axis gate tái sử dụng IOU

Với box detection `b` và box tham chiếu `r`, dùng:

```text
rho = (1-IOU)/(1+IOU)
distance(center(b),center(r)) < rho * max(width(b),height(b),width(r),height(r),1)
```

Thử cả prediction và observation cuối của track. Quan hệ rho xuất phát từ độ dịch một trục của hai box bằng kích thước tại ngưỡng IoU; việc dùng khoảng cách Euclidean và cạnh dài là một heuristic mở rộng. Không diễn giải nó là xác suất thống kê hoặc kiểm định Bayes đã hiệu chuẩn.

## Các tham số số học của bản chọn

| Tên | Giá trị | Vai trò |
|---|---:|---|
| DET |0.252139558052|High detection/association threshold|
| LOW |0.101281914078|Lower bound của tầng low confidence|
| NMS |0.6022380941698|NMS high detections trong tracker|
| IOU |0.3095489098032|Association gate và axis-gate-derived rho|
| MIN_HITS |3|High observations xác nhận tentative|
| MAX_AGE |160|Unexplained loss budget và giới hạn emission từ lần quan sát thật|
| HEAD_FRACTION |0.1843326829839|Tỉ lệ dải đầu box để tìm pixel bão hòa|
| EV |0.7233689491673|IoU tối thiểu với box quan sát cuối khi xuất prediction|
| INIT |0.4|Ngưỡng riêng cho birth, lấy ý tưởng init threshold của TrackTrack|

Không rút gọn bảng này thành “7 tham số”. Ngoài 9 số trên còn các lựa chọn cấu trúc, mask V/S/size, thông số Kalman/ORU của OC-SORT, input detector, NMS cache/cap và hậu xử lý output phải công bố.

Các lựa chọn còn lại:

```text
LIFECYCLE=true       ADMISSIBLE=false       BIRTH_POLICY=axis
NOISE=false          COAST_MODE=budget      MEASURE=posterior
LOW_POLICY=legacy    ASSOCIATION=iou        LOW_OUTPUT=all
OUTPUT_FILTER=none   RECOVERY=true          LOST_BIRTH=defer
```

`NOISE=false`: không dùng NSA/confidence-dependent measurement noise trong bản chọn. `LOW_POLICY=legacy`: low dùng IOU association, không dùng EV nghiêm ngặt. `LOW_OUTPUT=all`: low đã association thành công có thể xuất box bình thường. `own=false`: không bật cơ chế ownership suppression của bản HALO cũ. Không gọi bản này là `neg50` vì EV hiện tại khác 0,5 và cấu trúc đã thay đổi.

## Rủi ro và đánh đổi

Glare là bằng chứng môi trường dễ có false explanation. Đèn không thuộc người, glare kéo dài trên vùng trống, hoặc đối tượng đi ra khỏi cảnh vẫn có thể tạo prediction FP. EV và emission budget hạn chế điều này nhưng không loại hết. Axis gate có thể veto nhầm birth cạnh người khác; recovery duy nhất vẫn có thể nối sai khi một old track không phải cùng người nhưng hình học tình cờ phù hợp.

Không ReID giúp bỏ chi phí appearance inference và tránh feature bị glare/domain shift, nhưng làm ID recovery phụ thuộc hình học. MAX_AGE đo theo frame, nên thời lượng thực thay đổi theo FPS; không tự chuyển thành giây mà vẫn coi cấu hình tương đương. Các cảnh đông, giao cắt hoặc camera di chuyển mạnh cần đánh giá riêng.

## API Python tối thiểu

```python
import json
import numpy as np
from exsort.api import create_tracker
from exsort.cache import saturation

config = json.load(open("configs/selected.json", encoding="utf8"))
tracker = create_tracker(config)
# Một config/process; cấu hình nội bộ dùng module globals giống snapshot gốc.
# image_bgr: ảnh OpenCV HxWx3 uint8; dets: normalized cx,cy,w,h + score.
mask = np.unpackbits(saturation(image_bgr)).reshape(512, 512).astype(bool)
H, W = image_bgr.shape[:2]
dets = [{"box": np.array([.5, .5, .1, .4], np.float32), "conf": .9}]
tracks = tracker.step(dets, W, H, frame_idx=1, sat=mask)
for track in tracks:
    print(track.id, track.out_box())
```

Tham số frame_idx giữ theo API nghiên cứu; caller phải gọi mỗi frame đúng thứ tự và không bỏ qua frame không có detection. Mỗi sequence tạo tracker mới. Dùng process riêng khi thử nhiều cấu hình, không tạo hai tracker khác config rồi chạy xen kẽ trong cùng process.
