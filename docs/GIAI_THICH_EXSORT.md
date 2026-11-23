# Giải thích chi tiết Ex-Sort

Code: [exsort/tracker.py](../exsort/tracker.py) (toàn bộ tracker), [exsort/api.py](../exsort/api.py) (kiểm tra cấu hình, tạo tracker).
Cấu hình: [configs/final.json](../configs/final.json). Kiểm thử: [tests/test_pipeline.py](../tests/test_pipeline.py)
(`test_credibility_rules`, `test_glare_output_limit_and_memory`, `test_unique_and_ambiguous_recovery`).

---

## 1. Ex-Sort trong ba câu

Ex-Sort là tracker nhiều người không dùng appearance (không ReID), dựa trên Kalman của OC-SORT và ghép bằng IoU.
Ý tưởng chính: khi một người mất detection mà phần đầu box dự đoán chạm vùng chói (mask bão hòa), tracker coi lần mất
đó là "có lý do". Khi đó nó giữ ID lâu hơn và có thể xuất box dự đoán trong tối đa 1 giây.
Ex-Sort còn dùng ba luật "detection đáng tin" (score ≥ 0,6). Ba luật này không thêm tham số, chỉ dùng lại `INIT`, `MIN_HITS` và `MAX_AGE`.

---

## 2. Đầu vào

| Thành phần | Giá trị | Ghi chú |
|---|---|---|
| Detector | YOLOX-X, checkpoint epoch 15 | Chạy một lần, lưu cache |
| Cache detection | score ≥ 0,1; NMS 0,65; tối đa 100 box/frame | Ex-Sort và mọi baseline dùng chung cache này |
| Mask chói | resize 512×512, HSV, pixel có V ≥ 235 và S ≤ 89 | Tính từ ảnh, không học |
| FPS | FPS thật của từng sequence | Đổi giây → số frame |

---

## 3. Một frame được xử lý như thế nào

Thứ tự dưới đây đúng với 7 bước đánh số trong `ExSort.step` ([tracker.py](../exsort/tracker.py)).

**Bước 1: lấy detection.** Lấy mọi detection có score ≥ `DET` = 0,1, rồi chạy NMS của tracker với ngưỡng 0,6.
Mọi detection đi vào **một lượt ghép**, không chia high/low như ByteTrack.

> Detection có score 0,1–0,6 **không chỉ được lưu**. Nếu ghép được với một track đã có, nó **cập nhật Kalman** và box
> của track được xuất ở frame đó. Điều nó **không** làm được là **tạo track mới** (cần ≥ 0,6), và nó không giúp track
> mới được xác nhận (cần 3 detection ≥ 0,6).

**Bước 2: dự đoán.** Kalman OC-SORT dự đoán box cho mọi track. State 7 chiều `[cx, cy, area, ratio, vx, vy, v_area]`.

**Bước 3: ghép.** Hàm `match`, thuật toán Hungarian trên IoU, chấp nhận cặp có IoU ≥ `IOU` = 0,3.
- Ghép track đã xác nhận (confirmed) trước. Vòng 1 dùng box dự đoán. Vòng 2 dùng box detection ghép cuối của track
  (cách của OC-SORT), cho những track và detection còn dư.
- Sau đó ghép track tạm (tentative) với detection còn lại, cũng hai vòng như trên.
- Khi một track ghép lại sau khi mất dấu, ORU của OC-SORT nội suy lại trạng thái trong khoảng mất.

**Bước 4: track không ghép được.**
- Track confirmed: lấy dải trên cùng của box dự đoán, cao bằng 25% chiều cao box (`HEAD_FRACTION` = 0,25, hàm
  `glare_in_head_band`). Nếu dải này chứa ít nhất một pixel của mask → lần mất này **có lý do** (glare), bộ đếm
  `unexplained` giữ nguyên. Nếu không → `unexplained` tăng 1.
- Track tentative bị xóa ngay. Vì vậy 3 detection đáng tin để xác nhận phải đến khi track chưa mất frame nào.
- Track bị xóa khỏi bộ nhớ khi `unexplained > MAX_AGE` = 120 frame. Frame mất có glare không tính, nên người đứng
  trong vùng chói được giữ ID lâu hơn.

**Bước 5: sinh track mới (birth).** Detection còn dư chỉ sinh track nếu:
- score ≥ `INIT` = 0,6;
- không trùng với track **vừa ghép ở frame này**: IoM (giao chia diện tích box nhỏ hơn) với box ghép cuối < 0,3;
- không trùng với track **đang mất dấu**: khoảng cách tâm tới box dự đoán và tới box ghép cuối đều ≥ ρ × cạnh dài nhất,
  với ρ = (1 − IOU)/(1 + IOU) = 0,7/1,3 ≈ 0,54 (hàm `near`, "axis gate"). Nếu trùng, birth bị chặn để track cũ có cơ
  hội ghép lại.

**Bước 6: khôi phục ID (`recover_ids`).** Xét track mới có **đúng 3 lần ghép** (tính cả detection score thấp), vừa ghép
ở frame này và chưa từng được xác nhận. Nếu **đúng một** track cũ đã xác nhận, đang mất dấu, thỏa axis gate thì track mới
nhận ID cũ và track cũ bị bỏ. Có 0 hoặc ≥ 2 ứng viên thì giữ ID mới (tránh nối sai). Bước này chỉ chạy sau 3 frame đầu
của sequence.

**Bước 7: xuất box.** Một track được xuất khi đã confirmed và:
- **ghép được ở frame này** → xuất box Kalman sau cập nhật (posterior), hoặc
- **không ghép được** → chỉ xuất box dự đoán khi đủ **cả bốn** điều kiện (hàm `coasts`):
  1. vùng đầu đang chạm mask ở frame này (glare);
  2. chưa quá 1 giây kể từ lần ghép cuối (`GLARE_COAST_SECONDS` = 1);
  3. IoU(box dự đoán, box ghép cuối) ≥ `EV` = 0,9 (box dự đoán chưa trôi xa);
  4. trung bình score các detection đã ghép của track ≥ 0,6 (luật đáng tin số 2).
- Không có glare → **không xuất** box dự đoán. Code không có nhánh "coast mù".
- Luật đáng tin số 3: nếu detection đáng tin gần nhất đã cũ hơn 120 frame → không xuất box (track vẫn còn trong bộ nhớ).

Tóm tắt: **ID được giữ lâu nhờ glare; box chỉ được vẽ khi có detection hoặc khi glare giải thích được việc mất detection.**

---

## 4. Ba luật "detection đáng tin"

Một detection **đáng tin** khi score ≥ `INIT` = 0,6, tức cùng ngưỡng dùng để sinh track.

| # | Luật | Vấn đề nó sửa |
|---|---|---|
| 1 | Track chỉ confirmed khi có 3 detection đáng tin (bản trước: 3 detection bất kỳ ≥ 0,1) | Vật thể không phải người (đèn, máy móc) có score thấp vẫn đủ 3 lần ghép và trở thành track thật |
| 2 | Box dự đoán dưới glare chỉ xuất khi trung bình score các detection đã ghép ≥ 0,6 | Track "ma" có score trung bình thấp, đứng yên dưới đèn, được coast và sinh FP |
| 3 | Không xuất box khi detection đáng tin gần nhất cũ hơn 120 frame | Track sống mãi nhờ toàn detection score thấp (ghép với vật đứng yên), mỗi frame sinh FP |

Vì sao không thêm tham số: ngưỡng 0,6 là `INIT`, 120 là `MAX_AGE`, 3 là `MIN_HITS`, tất cả đã có sẵn.

Vì sao luật 3 chỉ tắt **output** mà không xóa track: trên val đã thử biến thể "chỉ detection đáng tin mới gia hạn sống
cho track" (xóa track). HOTA rơi từ 57,98 xuống **53,29** và IDF1 từ 77,13 xuống 66,06. Lý do: ở sequence
`400_sequence_morongmo` (78% GT của val), nhiều người thật chỉ có detection score thấp trong hơn 120 frame. Xóa track
làm mất ID của họ. Tắt output thì giữ được ID, và khi họ có lại detection ≥ 0,6 thì box xuất lại với ID cũ.

---

## 5. Tham số

| Tham số | Giá trị | Miền tìm kiếm | Vai trò |
|---|---:|---|---|
| `DET` | 0,1 | Cố định | Score thấp nhất được dùng |
| `INIT` | 0,6 | 0,50–0,70 | Score của detection đáng tin |
| `NMS` | 0,6 | 0,30–0,65 | IoU để bỏ box trùng |
| `IOU` | 0,3 | 0,15–0,50 | Ngưỡng ghép, IoM khi birth và ρ của axis gate |
| `MIN_HITS` | 3 | 2–5 | Số detection đáng tin để xác nhận |
| `MAX_AGE` | 120 frame | 20–180 | Giới hạn miss không có glare |
| `HEAD_FRACTION` | 0,25 | 0,15–0,40 | Phần trên của box được kiểm tra glare |
| `EV` | 0,9 | 0,30–0,90 | IoU tối thiểu giữa box dự đoán và box ghép cuối |
| `GLARE_COAST_SECONDS` | 1,0 s | 1–4 | Thời gian coast tối đa |

Một lượt SMAC3 (tối ưu Bayes) trên val chọn cùng lúc tám tham số: 100 trial, seed 20261123, cost `200 − HOTA − MOTA`. Chi
tiết ở mục 11 và [TUNING.md](TUNING.md).

---

## 6. Chọn cấu hình trên val (ablation từng bước)

Tất cả trên val. Mỗi hàng thêm một thay đổi vào hàng trên.

| Bước | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Một lượt ghép, chưa có ba luật | 57,90 | 60,87 | 55,81 | 76,95 | 79,93 | 62 | 11 111 | 26 506 |
| + luật 2 (coast cần score TB ≥ 0,6) | 57,92 | 60,95 | 55,79 | 77,00 | 80,10 | 64 | 10 690 | 26 605 |
| + luật 1 (confirm cần 3 det ≥ 0,6) | 58,00 | 61,00 | 55,89 | 77,12 | 80,21 | 59 | 10 483 | 26 615 |
| **+ luật 3 (tắt output) = bản cuối** | **57,98** | **60,99** | **55,84** | **77,13** | **80,29** | **58** | **9 888** | **27 072** |

Các biến thể bị loại trên val:

| Biến thể | HOTA val | Vì sao loại |
|---|---:|---|
| Bỏ hẳn track có score TB < 0,6 (không chỉ coast) | 57,58 | Mất người thật có score thấp |
| Chỉ detection đáng tin gia hạn sống (xóa track) | 53,29 | Mất ID ở morongmo (mục 4) |
| Xuất box detection thô thay vì posterior Kalman | 57,93 | IDSW tăng lên 84 |
| Coast mù 0,2 / 0,5 / 1,0 s | 58,04 / 58,05 / 58,06 | Không dùng: box dự đoán chỉ được vẽ khi có glare (mục 8) |

---

## 7. Kết quả trên test (48 sequence, 114 097 box GT)

| Tracker | HOTA↑ | DetA↑ | AssA↑ | IDF1↑ | MOTA↑ | IDSW↓ | FP↓ | FN↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ex-Sort** | **63,07** | **62,76** | **64,08** | **81,79** | 80,09 | **33** | 4 304 | **18 382** |
| TrackTrack | 58,88 | 62,29 | 56,59 | 72,83 | **80,20** | 56 | 3 038 | 19 492 |
| HybridSORT | 57,59 | 61,63 | 54,76 | 70,76 | 79,28 | 87 | 3 393 | 20 161 |
| BoT-SORT | 56,28 | 62,12 | 52,01 | 68,50 | 79,72 | 95 | 2 618 | 20 422 |
| ByteTrack | 56,22 | 61,65 | 52,29 | 68,89 | 79,21 | 72 | 2 814 | 20 832 |
| DiffMOT | 56,01 | 61,97 | 51,64 | 67,97 | 79,53 | 111 | 3 210 | 20 037 |
| Deep-OC-SORT | 53,28 | 59,02 | 49,12 | 65,23 | 75,57 | 89 | 1 598 | 26 185 |
| OC-SORT | 53,27 | 58,93 | 49,17 | 65,50 | 75,55 | 97 | 1 603 | 26 194 |
| DIP | 51,28 | 61,15 | 44,25 | 58,05 | 77,10 | 258 | 7 028 | 18 845 |

Baseline: số từ bảng so sánh của nhóm, chạy code gốc với cấu hình mặc định MOT17 và FastReID trên cùng detection cache.
Các baseline không được tune trên val GlareMOT, còn Ex-Sort thì có.

So với TrackTrack:
- HOTA +4,19; AssA +7,49; IDF1 +8,96; IDSW 33 so với 56: lợi thế chính là **giữ ID**.
- DetA +0,47; FN ít hơn 1 110.
- MOTA thấp hơn 0,11 vì FP nhiều hơn 1 266. Tổng lỗi MOTA (FP + FN + IDSW): Ex-Sort 22 719, TrackTrack 22 586.
  Hai bên gần như bằng nhau về MOTA.

---

## 8. Các đánh đổi

**a) Vì sao xuất box từ detection score thấp mà FP không tăng mạnh?**
Một box thêm vào có precision p (xác suất là TP).
- MOTA = 1 − (FN + FP + IDSW)/GT. Thêm box: lỗi giảm p (bớt FN) và tăng 1 − p (thêm FP). Lợi khi 1 − 2p < 0, tức **p > 0,5**.
- DetA ≈ TP/(TP + FN + FP). Thêm box: tử tăng p, mẫu tăng 1 − p. Lợi khi **p > DetA/(1 + DetA) ≈ 0,37**.

Đo trên val: box xuất từ detection 0,1–0,6 có precision **0,70** (10 957 box). Vượt cả hai ngưỡng, nên giữ chúng làm tăng
cả DetA và MOTA. Box từ detection ≥ 0,6 có precision 0,96. Box coast dưới glare có precision 0,74.
(Các precision này đo trên bản trước luật 3; luật 3 chỉ bớt thêm box.)

**b) Vì sao không coast mù?**
Coast mù 1 s tăng val HOTA thêm 0,08 (57,98 → 58,06), nhỏ hơn mức chênh giữa nhiều biến thể khác. Không dùng nó giúp
method nhất quán: **chỉ glare mới là lý do để vẽ box khi không có detection**. Ngoài lúc có glare, luật xuất box giống
ByteTrack, BoT-SORT và TrackTrack (chỉ xuất track đang được ghép).

**c) Không ReID:** nhanh, không bị appearance hỏng vì chói, nhưng nối ID chỉ dựa vào hình học.

---

## 9. Giới hạn

1. **Detector bỏ sót:** trên test, khoảng 15 500 trong 18 382 FN không có detection nào với IoU ≥ 0,5. Tracker không tạo
   được box từ không có gì; muốn giảm phải cải thiện detector.
2. **Lệch vị trí box:** khoảng 1 958 FP trên test là box gần đúng (IoU 0,3–0,5 với GT). Lệch không có hướng cố định
   (val: box nhỏ hơn GT; test: box rộng hơn), nên không sửa bằng một hệ số được.
3. **IDSW còn lại** chủ yếu là hai người đi sát nhau đổi ID. Không có appearance thì khó tách.
4. **MAX_AGE tính theo frame**, nên thời gian thực phụ thuộc FPS.
5. **Mask chói** có thể giải thích nhầm (đèn trần, vùng trống sáng). EV 0,9, giới hạn 1 s và luật đáng tin hạn chế điều này
   nhưng không loại hết.

---

## 10. Câu hỏi thường gặp

**H: Dùng detection score 0,1 thì không sợ FP à?**
Đ: Detection thấp chỉ được dùng để **nối tiếp** track đã có, không tạo track mới, không xác nhận track. Box từ chúng
có precision 0,70 trên val, trên ngưỡng hòa vốn 0,5 của MOTA và 0,37 của DetA. Trên val, bỏ detection dưới 0,60
làm HOTA giảm 2,78 và MOTA giảm 1,96.

**H: Tham số được chọn thế nào? Có tune trên test không?**
Đ: Mọi lựa chọn dựa trên val. Một lượt SMAC3 gồm 100 trial tối thiểu `200 − HOTA − MOTA`; cấu hình có tổng HOTA + MOTA cao
nhất được chốt rồi chạy test một lần.

**H: Ba luật đáng tin có phải là heuristic thêm vào cho đẹp số không?**
Đ: Chúng không có tham số mới; dùng lại ngưỡng birth, `MIN_HITS` và `MAX_AGE`. Ý tưởng chung là một track phải có bằng chứng
"đây là người" ở cùng mức với lúc sinh track. Mỗi luật được giữ vì làm tăng tổng HOTA + MOTA trên val (bảng mục 6).

**H: Sao không xóa track khi không có detection đáng tin?**
Đ: Đã thử trên val: HOTA rơi xuống 53,29 vì người thật ở morongmo thường chỉ có detection thấp lâu. Tắt output giữ được ID.

**H: Glare giúp gì cụ thể?**
Đ: (1) frame mất có glare không tính vào tuổi track nên ID sống lâu hơn; (2) cho phép vẽ box dự đoán tối đa 1 s.
Trên test, Ex-Sort có AssA và IDF1 cao nhất bảng (AssA 64,08 so với 56,59 của TrackTrack), dù không dùng ReID.
Trên val, tắt mask glare làm HOTA giảm 2,79, AssA giảm 4,69 và IDF1 giảm 5,37. Phần lớn đến từ (1): nếu glare không còn ngừng
việc tính tuổi track thì HOTA giảm 2,64; nếu chỉ bỏ box dự đoán thì HOTA giảm 0,19.

**H: Sao MOTA vẫn thua TrackTrack?**
Đ: Chênh 0,11 điểm. Ex-Sort xuất thêm box để giảm FN và giữ ID, nên FP cao hơn 1 266 nhưng FN thấp hơn 1 110 và
IDSW thấp hơn 23. Tổng lỗi hai bên chỉ khác 133 trên 114 097 box GT.

**H: Baseline có được tune không?**
Đ: Không; dùng cấu hình mặc định của repo gốc (MOT17) với FastReID bản MOT20. Đây là giới hạn của so sánh.

---

## 11. Tiêu chí chọn trên val

**Tiêu chí: tổng HOTA + MOTA trên val cao nhất**, vì đây là hai metric chính. Cost mà SMAC3 tối thiểu là `200 − HOTA − MOTA`.

- Tám cấu hình Sobol và một cấu hình khởi đầu (từ các lần chạy sơ bộ trên cùng tập val) chạy trước, SMAC3 đề xuất 91 cấu hình
  còn lại. Mỗi trial chạy toàn bộ 69 612 frame val.
- Cấu hình khởi đầu có cost thấp nhất trong 100 trial và được chọn: HOTA 57,98, MOTA 80,29, IDF1 77,13, 58 IDSW.
- Cấu hình tốt nhất do SMAC3 đề xuất: HOTA 57,98, MOTA 80,26, với `MIN_HITS` 2 và glare 1,03 s; các giá trị khác gần như trùng.

Các bản có coast mù (mục 6) không nằm trong miền tìm kiếm: theo thiết kế, box dự đoán chỉ được vẽ khi glare giải thích việc mất
detection.

Phân tích lỗi (track ma, detection score thấp) dẫn tới ý tưởng của ba luật. Ba luật không có tham số và được giữ chỉ dựa trên
số val.
