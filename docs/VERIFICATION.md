# Kiểm tra bản phát hành

## Tracker cuối

Module [exsort/tracker.py](../exsort/tracker.py) được viết lại từ code nghiên cứu, chỉ giữ hành vi của cấu hình cuối. Replay trên cùng cache YOLOX-X cho file dự đoán MOT **giống từng byte** với code nghiên cứu: 71/71 sequence val và 48/48 sequence test.

| Tập | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Val, 69.612 frame | 57,98 | 60,99 | 55,84 | 77,13 | 80,29 | 58 | 9.888 | 27.072 |
| Test, 118.632 frame | 63,07 | 62,76 | 64,08 | 81,79 | 80,09 | 33 | 4.304 | 18.382 |

Bảng trên được chạy với numpy 1.26.4 và scipy 1.15.3 ([requirements-tested.txt](../requirements-tested.txt)). Với numpy 2.2.6, test cho FP 4.303 và FN 18.381.

## Bộ test CPU

`python -m unittest discover -s tests -v` chạy trên dữ liệu tổng hợp, không cần dataset hay weight:

- resume từ checkpoint sau một block, kể cả khi có đoạn ghi chưa commit, cho file dự đoán cùng SHA với lần chạy liên tục;
- `tune` nhỏ trên val, khóa selection, đánh giá test, chạy lại không đổi kết quả; từ chối tune trên test và từ chối selection bị sửa;
- phát hiện GT bị thay đổi; mask tính từ pixel thật; chấm lại khi GT đổi mà không replay;
- kiểm tra khóa cấu hình và FPS;
- khôi phục ID khi có đúng một ứng viên, giữ ID mới khi có hai; track đang mất chặn track mới ở gần;
- box dự đoán chỉ xuất ở frame có glare, trong 1 giây kể từ lần ghép cuối, tại 19, 22 và 25 FPS;
- ba luật detection đáng tin: xác nhận, xuất box dưới glare, ngừng xuất sau `MAX_AGE` frame không có detection đáng tin;
- mọi trường HOTA/CLEAR/Identity/Count khớp TrackEval chạy từ file.

## Detector

Các kiểm tra dưới đây chạy với opencv-python 5.0.0.93.

- Preprocessing FP16 khớp từng phần tử với `ValTransform` của fork TrackTrack/ByteTrack trên ảnh thật (`scripts/check_yolox_transform.py`).
- Adapter YOLOX-X chạy GPU trên 4 frame thật, so với cache val: offsets, mask và box giống nhau; confidence lệch tối đa khoảng 0,00049. Inference mới vì vậy không bảo đảm giống từng bit với cache cũ; dùng chính cache cũ khi cần lặp lại đúng input.

```powershell
python -m unittest discover -s tests -v
python scripts/check_yolox_transform.py --yolox-root "third_party/TrackTrack/1. YOLOX" --image PATH_TO_IMAGE
```

CI chỉ chạy bộ test CPU.
