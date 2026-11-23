# Ex-Sort

**Ex-Sort** là bộ theo dõi nhiều người cho video có glare (vùng sáng bão hòa do đèn). Ex-Sort dùng Kalman của OC-SORT và ghép IoU, không dùng ReID. Khi một người mất detection, Ex-Sort kiểm tra mask glare ở vùng đầu của box dự đoán: nếu có glare thì lần mất được coi là do glare, track không bị tính tuổi và có thể xuất box dự đoán trong tối đa 1 giây. Detection có score thấp vẫn giữ track, nhưng chỉ detection đáng tin (score ≥ 0,6) mới tạo và xác nhận track.

Repo gồm tracker, cấu hình cuối, tạo cache detection/mask, replay trên val, tìm tham số bằng SMAC3 trên val, đánh giá test bằng TrackEval.

## Kết quả trên GlareMOT

Detector YOLOX-X (epoch 15), [cấu hình cuối](configs/final.json), TrackEval. Chỉ số chất lượng tính theo %.

| Tập | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Val (71 sequence, 69.612 frame) | 57,98 | 60,99 | 55,84 | 77,13 | 80,29 | 58 | 9.888 | 27.072 |
| Test (48 sequence, 118.632 frame, 114.097 box GT) | 63,07 | 62,76 | 64,08 | 81,79 | 80,09 | 33 | 4.304 | 18.382 |

Cấu hình được chọn trên val rồi đánh giá một lần trên test. [Cách chọn tham số](docs/TUNING.md)

## Phương pháp

1. Giữ detection có score ≥ 0,1, NMS 0,6.
2. Kalman dự đoán mọi track. Ghép track confirmed bằng Hungarian trên IoU (ngưỡng 0,3): trước với box dự đoán, sau với box ghép cuối. Rồi ghép track tentative với detection còn dư.
3. Track tentative mất detection bị xóa. Track confirmed mất detection: nếu mask glare chạm 25% phía trên box dự đoán thì lần mất được glare giải thích. Track bị xóa sau hơn 120 lần mất không có glare.
4. Detection còn dư tạo track khi score ≥ 0,6 và không bị track nào chiếm. Track được xác nhận sau 3 detection đáng tin.
5. Track mới vừa xác nhận nhận lại ID của track cũ nếu có đúng một track đang mất ở gần.
6. Xuất box Kalman cho track confirmed có detection ghép. Track đang mất chỉ xuất box dự đoán khi frame hiện tại có glare ở vùng đầu, chưa quá 1 giây từ lần ghép cuối, box dự đoán còn khớp box ghép cuối (IoU ≥ 0,9) và trung bình score các detection đã ghép ≥ 0,6. Track không có detection đáng tin trong hơn 120 frame thì không xuất box nhưng vẫn giữ ID.

Chi tiết từng bước, tham số và API Python: [docs/METHOD.md](docs/METHOD.md). Giải thích dài hơn, kèm ablation trên val và các đánh đổi: [docs/GIAI_THICH_EXSORT.md](docs/GIAI_THICH_EXSORT.md).

```python
from exsort.api import create_tracker
from exsort.io import read

tracker = create_tracker(read("configs/final.json"), fps=25)   # một tracker cho mỗi sequence
tracks = tracker.step(dets, W, H, sat=mask)                     # gọi mỗi frame, kể cả frame trống
```

## 1. Cài đặt

Hướng dẫn dùng PowerShell và Python 3.10. Các lệnh `python -m exsort ...` cũng chạy trên Linux.

```powershell
git clone https://github.com/minhdq245/exsort.git
cd exsort
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -c requirements-tested.txt -e ".[tune]"
python -m pip install -r requirements-eval.txt
python -m unittest discover -s tests -v
```

TrackEval được ghim ở commit `12c8791b303e0a0b50f753af204249e622d0281a`, SMAC3 ở `2.4.1`. Replay, tune và đánh giá chỉ cần CPU. Tạo cache detection cần thêm PyTorch, torchvision và YOLOX.

## 2. Dữ liệu và split

```text
GlareMOT/
  train/<sequence>/images/frame_000000.jpg ...
  train/<sequence>/mot/gt/gt.txt
  test/<sequence>/images/frame_000000.jpg ...
  test/<sequence>/mot/gt/gt.txt
```

`gt.txt` theo định dạng MOT 9 cột `frame,id,x,y,w,h,mark,class,visibility`, frame bắt đầu từ 1, box theo pixel gốc, `class=1` là người, `mark=0` bị bỏ khi chấm.

Split nằm trong [configs/split_seed20261024.json](configs/split_seed20261024.json): 71 sequence train, 48 sequence test. **Val** là nửa sau của từng sequence train: với N ảnh, bỏ N//2 ảnh đầu, đánh lại frame từ 1 và dùng tracker mới. Detector chỉ được train trên nửa đầu các sequence train. Val và test giữ **mọi frame**, kể cả frame trống hoặc toàn glare.

```powershell
$Data = "D:\datasets\GlareMOT"
python -m exsort prepare --data "$Data" --split configs/split_seed20261024.json --role val --output runs/manifests/val.json
python -m exsort prepare --data "$Data" --split configs/split_seed20261024.json --role test --output runs/manifests/test.json
```

Manifest ghi tên và thứ tự ảnh, kích thước, SHA256 của GT và FPS (từ `meta/DONE.json`, trường `output_integer_fps`, hoặc `seqinfo.ini`, trường `frameRate`). Thêm `--hash-images` để băm mọi ảnh.

## 3. Tạo cache detection và mask

Detector là YOLOX-X person-only của fork TrackTrack/ByteTrack (không phải Ultralytics). Repo không chứa weight hay dataset; đặt checkpoint vào `weights/`.

```powershell
git clone https://github.com/kamkyu94/TrackTrack.git third_party/TrackTrack
git -C third_party/TrackTrack checkout ee7f1c5fcbdcac48ed8bfab38d52c0006bf304da
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-yolox.txt
python -m exsort cache --manifest runs/manifests/val.json --yolox-root "third_party/TrackTrack/1. YOLOX" --exp examples/yolox_x_person.py --weights weights/yolox_x_glaremot_epoch15.pth --output cache/val --batch-size 4 --workers 4
python -m exsort verify-cache --manifest runs/manifests/val.json --cache cache/val
```

Làm tương tự cho test (`--manifest runs/manifests/test.json --output cache/test`). Chọn bản torch/CUDA hợp với máy theo [hướng dẫn PyTorch](https://pytorch.org/get-started/locally/).

| Thành phần | Giá trị |
|---|---|
| Input detector | 800 × 1440 (cao × rộng), resize giữ tỉ lệ, pad màu 114 |
| Chuẩn hóa | BGR→RGB; mean `.485,.456,.406`; std `.229,.224,.225` |
| Precision | FP16, không fuse Conv/BN |
| Score | objectness × xác suất lớp person |
| Score floor / NMS / tối đa | 0,001 / 0,65 / 100 detection mỗi frame |
| Lưu box | `x1,y1,x2,y2,score`, pixel gốc |
| Mask glare | Resize về 512 × 512 (INTER_AREA), HSV, `V ≥ 235` và `S ≤ 89`, lưu `packbits` |

Cache lưu theo block 512 frame, mỗi block có SHA256. Vì cache đã áp NMS 0,65 và score floor 0,001, tracker cần `NMS ≤ 0,65` và `DET ≥ 0,001`.

## 4. Replay trên val

```powershell
python -m exsort validate --manifest runs/manifests/val.json --cache cache/val --config configs/final.json --output runs/val_final --workers 8
```

Lệnh replay từng sequence, ghi dự đoán MOT, chấm HOTA/CLEAR/Identity/Count rồi khóa cấu hình vào `runs/val_final/selection.json`.

```text
runs/val_final/
  protocol.json
  metrics.json                     # tổng và từng sequence
  all_trackeval_metrics.json       # toàn bộ trường của TrackEval
  selection.json                   # cấu hình + chữ ký split/source/detector
  trackers/exsort/data/<seq>.txt   # dự đoán MOT
  work/<seq>/checkpoint.pkl.gz     # trạng thái để chạy tiếp
  complete.json
```

## 5. Tìm tham số trên val (tùy chọn)

```powershell
python -m exsort tune --manifest runs/manifests/val.json --cache cache/val --output runs/tune --workers 8
```

SMAC3 đề xuất tuần tự, mỗi trial replay toàn bộ val, chọn cấu hình có tổng HOTA + MOTA val cao nhất. Lệnh không đọc test. [Chi tiết](docs/TUNING.md)

## 6. Đánh giá test

```powershell
python -m exsort evaluate --manifest runs/manifests/test.json --cache cache/test --selection runs/val_final/selection.json --output runs/test_final --workers 8
```

Lệnh kiểm tra selection chưa bị sửa, cùng split, cùng source code và cùng checkpoint detector với val. Đổi code hoặc cấu hình sau khi khóa thì phải chạy lại val vào thư mục mới.

- `metrics.json`: HOTA, DetA, AssA, IDF1, MOTA, IDSW, FP, FN tổng và từng sequence. HOTA tổng dùng `combine_sequences` của TrackEval; CLEAR và Identity dùng IoU 0,5.
- `trackers/exsort/data/*.txt`: `frame,id,x,y,w,h,1,-1,-1,-1`, frame từ 1, pixel gốc. Val ghi 2 chữ số thập phân, test 4.

## 7. Tạm dừng và chạy tiếp

```powershell
New-Item runs/val_final/STOP -ItemType File   # dừng sau block đang chạy
Remove-Item runs/val_final/STOP               # rồi chạy lại đúng lệnh cũ
```

STOP cũng dùng cho thư mục `cache` và `tune`. Ctrl+C cũng chạy tiếp được từ block hoặc trial đã lưu. Không mở hai tiến trình ghi cùng một thư mục output.

## 8. GT thay đổi nhưng ảnh giữ nguyên

```powershell
python -m exsort prepare --data "$Data" --split configs/split_seed20261024.json --role test --output runs/manifests/test_gt_v2.json
python -m exsort rescore --manifest runs/manifests/test_gt_v2.json --original-manifest runs/manifests/test.json --predictions runs/test_final/trackers/exsort/data --output runs/rescore_gt_v2 --workers 8
```

Chỉ chấm lại dự đoán đã có. Hai manifest phải có cùng ảnh và thứ tự frame.

## Cấu trúc repo

```text
exsort/tracker.py        # Ex-Sort
exsort/api.py            # kiểm tra cấu hình, tạo tracker
exsort/vendor/ocsort/    # Kalman, ORU và hàm ghép của OC-SORT
exsort/cache.py          # inference YOLOX-X, mask glare, cache
exsort/dataset.py        # manifest, split, GT
exsort/runner.py         # replay, checkpoint, khóa selection, đánh giá test, chấm lại
exsort/scoring.py        # TrackEval
exsort/tuning.py         # SMAC3 trên val
configs/final.json       # cấu hình cuối
configs/search.json      # miền tìm kiếm mặc định của tune
configs/split_seed20261024.json
docs/                    # phương pháp, chọn tham số, kiểm tra
tests/                   # test CPU trên dữ liệu tổng hợp
```

Xây dựng từ [OC-SORT](https://github.com/noahcao/OC_SORT), [ByteTrack](https://github.com/ifzhang/ByteTrack), [TrackTrack](https://github.com/kamkyu94/TrackTrack), [FilterPy](https://github.com/rlabbe/filterpy), [SMAC3](https://github.com/automl/SMAC3) và [TrackEval](https://github.com/JonathonLuiten/TrackEval). Giấy phép các thành phần này: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Kết quả kiểm tra bản phát hành: [docs/VERIFICATION.md](docs/VERIFICATION.md).
