# exsort

**exsort** là bộ theo dõi người dùng chuyển động, detection confidence thấp và bằng chứng chói sáng để xử lý mất detection. Bản được chọn **không dùng ReID**, không cập nhật tâm người theo tâm đèn. Repo chứa thuật toán mới nhất, cấu hình đã chọn, cache detector/mask, Bayesian tuning trên val và đánh giá bằng TrackEval trên test.

Tên method: **exsort**. HALO-SORT là tên dự án nghiên cứu trước đây.

```text
GlareMOT/images + person detector
              |
              v
  cache detection + saturation mask  (GPU một lần)
              |
              v
  replay nhiều cấu hình trên val     (CPU, không inference lại)
              |
              v
  chọn theo HOTA với ràng buộc MOTA
              |
              v
  selection.json được đóng băng
              |
              v
  một cấu hình trên test + TrackEval + MOT predictions
```

Hướng dẫn dưới đây dùng PowerShell và Python 3.10. Các lệnh `python -m exsort ...` cũng chạy trên Linux; thay đường dẫn và cách kích hoạt môi trường.

## 1. Clone và cài môi trường

```powershell
git clone https://github.com/minhdq245/exsort.git
cd exsort
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -c requirements-tested.txt -e ".[tune]"
python -m pip install -r requirements-eval.txt
python -m exsort --help
python -m unittest discover -s tests -v
```

Nếu PowerShell chặn activate, có thể gọi trực tiếp `.\.venv\Scripts\python.exe` thay cho `python`. Không cần đổi policy toàn máy.

TrackEval được ghim ở commit `12c8791b303e0a0b50f753af204249e622d0281a`; SMAC3 ở `2.4.1`. Các constraint đã kiểm tra nằm trong `requirements-tested.txt`. Bộ test nhỏ kiểm tra resume, đóng băng cấu hình, từ chối GT/config bị thay, hồi phục ID và parity với TrackEval file-based. Đây là kiểm tra phần mềm; không thay thế benchmark trên dữ liệu thật.

Chỉ cần CPU để dùng cache có sẵn, tune và đánh giá tracking. Bước tạo detection cache mới cần thêm PyTorch/torchvision và detector.

## 2. Chuẩn bị dữ liệu và split

```text
GlareMOT2 full/
  split.json
  train/
    <sequence>/
      images/frame_000000.jpg ...
      mot/gt/gt.txt
  test/
    <sequence>/
      images/frame_000000.jpg ...
      mot/gt/gt.txt
```

`gt.txt` có 9 cột MOT: `frame,id,x,y,w,h,mark,class,visibility`. Frame GT bắt đầu từ **1**, dù tên ảnh có thể bắt đầu từ 0. Box dùng pixel gốc, `class=1` là person. `mark=0` bị loại khi scoring. File `labels`/`labels_all` dùng cho annotation/detector không phải input trực tiếp của tracker.

Checkpoint detector phải được train chỉ bằng dữ liệu cho phép ở train (trong protocol temporal này: nửa đầu các sequence train, cộng external training data đã khai báo); không dùng ảnh/nhãn tracker val hoặc test để fine-tune detector. Repo này chỉ cung cấp cache/tuning/eval tracker, không chạy lại quá trình train detector.

Split tham chiếu nằm trong `configs/split_seed20261024.json`: **71 train /48 test**, seed `20261024`; `400_sequence_morongmo` chỉ ở train, `401_sequence_nga3` chỉ ở test. Có thể dùng `split.json` gốc của dataset; CLI bỏ qua các đường dẫn ổ đĩa cũ trong đó và lấy sequence theo tên. Không tự random split lại.

**Tracker val** lấy toàn bộ nửa sau của từng sequence train: với N ảnh, bỏ N//2 ảnh đầu, giữ N−N//2 ảnh sau, đánh lại frame từ 1 và khởi tạo tracker mới. Bản dữ liệu tham chiếu có **69.612 frame val**. Test dùng toàn bộ **118.632 frame/48 sequence**. Nếu ảnh hoặc timeline của bản dataset bạn dùng khác, số lượng sẽ khác và cần báo đúng số thực tế.

Tracker val/test phải giữ **đầy đủ frame**, kể cả frame trống, frame chói và frame không có người. Quy tắc lấy 2,5 FPS/lọc box chói dùng khi train detector là một quy trình khác, không áp dụng vào benchmark tracking.

```powershell
$Data = "D:\datasets\GlareMOT2 full"
python -m exsort prepare --data "$Data" --split configs/split_seed20261024.json --role val --output runs/manifests/val.json
```

Manifest ghi tên ảnh, thứ tự, kích thước, mtime, hash ảnh đầu/cuối và SHA256 GT. Thêm `--hash-images` nếu cần SHA256 của **mọi ảnh** để audit nghiêm ngặt; bước này đọc toàn bộ ảnh và tốn I/O. Chế độ mặc định không chứng minh nội dung mọi ảnh nếu ai đó thay file nhưng cố tình giữ nguyên size/mtime.

Nên chuẩn bị manifest test sau khi đã chốt val ở bước 7.

## 3. Tạo cache detector và mask một lần

### 3a. Cài detector đúng fork

Checkpoint nghiên cứu dùng YOLOX-X của fork **TrackTrack/ByteTrack**, person-only. Đây không phải model Ultralytics YOLO. Đặt checkpoint tin cậy của bạn trong `weights/`; repo không chứa weight hoặc dataset.

```powershell
git clone https://github.com/kamkyu94/TrackTrack.git third_party/TrackTrack
git -C third_party/TrackTrack checkout ee7f1c5fcbdcac48ed8bfab38d52c0006bf304da
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-yolox.txt
python -c "import torch,torchvision; print(torch.__version__,torchvision.__version__,torch.cuda.is_available())"
```

Lệnh CUDA 12.8 trên là ví dụ. Chọn cặp torch/torchvision và CUDA phù hợp driver/GPU qua [hướng dẫn PyTorch](https://pytorch.org/get-started/locally/). Môi trường kiểm tra adapter dùng torch `2.11.0+cu128` và torchvision `0.26.0+cu128`; các bản khác có thể cho khác biệt số học/NMS. Không dùng `--fuse` hay TensorRT khi cần đối chiếu cache cũ.

`examples/yolox_x_person.py` khai báo đúng architecture: depth=1,33, width=1,25, một class. Nếu checkpoint dùng architecture khác, cung cấp `--exp` tương ứng. Load weight dùng strict state dict và không tự bỏ key bị sai.

### 3b. Cache val

```powershell
python -m exsort cache --manifest runs/manifests/val.json --yolox-root "third_party/TrackTrack/1. YOLOX" --exp examples/yolox_x_person.py --weights weights/yolox_x_glaremot2_half_epoch15_fp16.pth --output cache/val --batch-size 4 --workers 4
python -m exsort verify-cache --manifest runs/manifests/val.json --cache cache/val
```

Thiết lập cố định:

| Thành phần | Giá trị |
|---|---|
| Input detector | **H×W =800×1440**, một lần resize giữ tỉ lệ |
| Padding | Màu 114, ảnh ở góc trên trái |
| Chuẩn hóa | BGR→RGB; mean `.485,.456,.406`; std `.229,.224,.225` |
| Precision | FP16 mặc định; không fuse Conv/BN |
| Score | objectness × person confidence |
| Confidence floor | 0,001 |
| NMS của cache | 0,65 |
| Giới hạn | 100 detection/frame sau NMS |
| Tọa độ lưu | `x1,y1,x2,y2,score`, pixel ảnh gốc |
| Saturation mask | Resize ảnh gốc về 512×512 bằng INTER_AREA, rồi HSV: **V≥235 và S≤89** |
| Mask lưu | `packbits`, 32.768 byte/frame; đo từ ảnh thật |

Mask dùng ngưỡng `S≤89` để khớp thí nghiệm cũ, không tự đổi thành 90. Không dùng mask giả toàn 0. Detection cache và mask nằm cùng block `.npz`, có offsets cho từng frame và SHA256 từng block. Frame không có detection vẫn có offsets/mask.

**Cache đã NMS 0,65/cap100 không thể khôi phục box bị bỏ.** Vì vậy tune phải có NMS≤0,65 và LOW≥0,001. Muốn mở rộng miền này phải tạo loại cache mới và công bố lại protocol.

### 3c. Chạy nhanh mà vẫn giữ protocol

- Một model GPU chạy batch; nhiều CPU worker đọc/resize ảnh. Bắt đầu batch=4, workers=4. Có thể đo throughput rồi thử batch=8 nếu VRAM cho phép. Không chạy 2–3 model song song trên cùng GPU một cách mặc định.
- Batch/frame loading chỉ tăng throughput; association vẫn theo đúng thứ tự frame từng sequence.
- Có lỗi CUDA OOM khi forward, adapter giảm đôi batch và chạy lại từ block đã lưu gần nhất. OOM ngay lúc nạp model cần giảm input/model hoặc giải phóng VRAM; thay input size là thay protocol.
- Block mặc định 512 frame. Cache lưu bằng NPZ không nén để giảm CPU; mask đã đóng gói bit. Nên đặt cache trên SSD và tránh đọc ảnh, ghi cache và nén dataset đồng thời trên một HDD.
- Batch/worker tối ưu phụ thuộc GPU, CPU và đĩa. Không có ETA chung cho mọi máy; đo từ `progress.json` và tốc độ thực tế.

### 3d. Nếu đã có cache từ thí nghiệm cũ

Không cần chạy lại GPU. Xem [hướng dẫn import cache cũ](docs/LEGACY_CACHE.md). Cần xác nhận ảnh/timeline còn đúng; cache cũ không có hash nội dung đầy đủ của mọi ảnh. GT đã sửa vẫn dùng lại detection nếu ảnh và timeline không đổi.

## 4. Kiểm tra một cấu hình trên val

```powershell
python -m exsort validate --manifest runs/manifests/val.json --cache cache/val --config configs/selected.json --output runs/val_selected --workers 8
```

Lệnh replay liên tục từng sequence, ghi MOT predictions, chạy HOTA/CLEAR/Identity/Count, rồi đóng băng `runs/val_selected/selection.json`. Đây là cách chạy cấu hình đã công bố nếu không cần tìm tham số mới. Không dùng số liệu cũ để mặc định rằng GT mới sẽ cho cùng điểm.

Output chính:

```text
runs/val_selected/
  protocol.json
  metrics.json                      # summary và từng sequence
  all_trackeval_metrics.json         # toàn bộ field/array của các metric
  selection.json                    # config + source/split/model signatures
  trackers/exsort/data/<seq>.txt     # MOT predictions
  work/<seq>/checkpoint.pkl.gz       # trạng thái resume sau mỗi cache block
  complete.json
```

## 5. Bayesian tuning chỉ trên val

```powershell
python -m exsort tune --manifest runs/manifests/val.json --cache cache/val --initial-config configs/selected.json --baseline configs/baseline.json --search configs/search_local.json --output runs/tune_local --workers 8
```

Lệnh thực hiện:

1. Kiểm tra cache, ảnh và GT; baseline được đo lại trên **val hiện tại**.
2. Lấy MOTA baseline làm ngưỡng tối thiểu. Không tái sử dụng giá trị floor từ GT cũ.
3. SMAC3 dùng random-forest surrogate và Log Expected Improvement để đề xuất trial. Các điểm Sobol/điểm khởi đầu tạo thông tin ban đầu; sau đó mỗi trial nhận feedback trước đề xuất tiếp theo.
4. Mỗi trial replay **toàn bộ val**, không đánh giá vài sequence rồi suy thành full score. Các sequence chạy CPU song song, một cấu hình tại một thời điểm để feedback rõ ràng.
5. Chọn HOTA pooled cao nhất trong các cấu hình đạt MOTA floor. Hòa điểm: IDF1, rồi MOTA, rồi ít IDSW hơn. Baseline luôn có trong tập ứng viên.
6. Lưu runhistory, từng trial, từng sequence và `selection.json` cuối cùng. **Không đọc manifest test trong tuning.**

Miền local đã dùng ở vòng nghiên cứu gần nhất:

| Tham số | Min | Max |
|---|---:|---:|
| DET |0,2181218260059|0,2781218260059|
| LOW |0,075|0,13|
| NMS |0,55|0,65|
| IOU |0,27|0,32|
| EV |0,55|0,78|
| MAX_AGE |60|200|

Mặc định local: **24 trial**, seed `20261030`; HEAD_FRACTION/MIN_HITS và cấu trúc lấy từ initial config. `configs/search_broad.json` có miền rộng 8 biến, 64 trial, seed `20261028`, phù hợp khảo sát lại. Đổi search file hoặc số trial phải dùng output mới. Đây là tối ưu ngân sách hữu hạn, không bảo đảm tối ưu toàn cục.

**Phân biệt với lịch sử:** CLI mới tune số trên một cấu trúc cố định. Lượt nghiên cứu cũ còn chọn cấu trúc bằng ablation, dùng batch 2–3 proposal và local checks. Vì vậy chạy CLI mới không bảo đảm sinh lại đúng thứ tự 84 ứng viên cũ. Danh sách ứng viên/điểm cũ nằm ở `history/validation_candidates_20261029.json`; quy trình lịch sử được mô tả trong [TUNING.md](docs/TUNING.md).

## 6. Ablation cơ chế, rồi chốt trước test

Nếu cần xác định cơ chế nào giúp, dùng các biến thể khai báo sẵn:

```powershell
python -m exsort ablate --manifest runs/manifests/val.json --cache cache/val --config configs/selected.json --variants configs/ablations.json --output runs/ablations --workers 8
```

Lệnh so sánh bỏ recovery, bật lost-birth veto, box raw, bỏ xuất dự đoán khi mất dấu, dùng chung DET cho birth và association low nghiêm ngặt. Các ứng viên được đánh giá trên đầy đủ val. Chọn theo cùng HOTA/MOTA-floor rule; ghi `best_config.json` và `selection.json`.

Có thể lấy `best_config.json` làm `--initial-config` cho một lượt tune mới; cần định trước ngân sách và ghi đầy đủ các lượt đã thử. Nếu cấu hình nằm ngoài miền search hiện tại, sửa miền search tương ứng trước khi chạy. Sau khi đã xem test, không dùng test để chọn vòng tune/biến thể tiếp theo rồi vẫn gọi đó là hold-out hoàn toàn mới.

## 7. Đóng băng cấu hình và eval test

Chọn đúng **một** `selection.json` từ val (ví dụ `runs/tune_local/selection.json`), rồi:

```powershell
python -m exsort prepare --data "$Data" --split configs/split_seed20261024.json --role test --output runs/manifests/test.json
python -m exsort cache --manifest runs/manifests/test.json --yolox-root "third_party/TrackTrack/1. YOLOX" --exp examples/yolox_x_person.py --weights weights/yolox_x_glaremot2_half_epoch15_fp16.pth --output cache/test --batch-size 4 --workers 4
python -m exsort verify-cache --manifest runs/manifests/test.json --cache cache/test
python -m exsort evaluate --manifest runs/manifests/test.json --cache cache/test --selection runs/tune_local/selection.json --output runs/test_frozen --workers 8
```

Nếu đã import cache test ở bước 3d, bỏ lệnh tạo cache GPU và dùng thư mục cache đã import.

Eval kiểm tra seal cấu hình, source signature, split, checkpoint detector giống val/test và SHA các đầu vào. Nếu thay code hoặc sửa cấu hình sau freeze, phải chạy lại val/freeze trong run mới. Seal là kiểm tra nhất quán, không phải chữ ký số chống giả mạo.

Kết quả:

- `metrics.json`: HOTA, DetA, AssA, IDF1, MOTA, IDSW, FP, FN tổng và từng sequence.
- `all_trackeval_metrics.json`: đầy đủ HOTA/CLEAR/Identity/Count, gồm các field phụ và các giá trị theo ngưỡng HOTA. HOTA dùng bộ ngưỡng mặc định TrackEval; CLEAR/Identity dùng IoU 0,5.
- `trackers/exsort/data/*.txt`: format `frame,id,x,y,w,h,score,-1,-1,-1`. Frame 1-based, pixel ảnh gốc. Score output=1 theo runner nghiên cứu; score detector thật vẫn giữ trong cache, không coi score=1 là confidence đã hiệu chuẩn.
- `frozen_selection.json`, `protocol.json`, `complete.json`: provenance để đối chiếu.

Metric quality hiển thị theo **%**; IDSW/FP/FN là số đếm. HOTA tổng dùng `combine_sequences` của TrackEval, không phải trung bình đơn giản HOTA từng sequence. Val dùng box GT/prediction 2 chữ số thập phân để khớp protocol cũ; test prediction 4 chữ số. `DO_PREPROC=False` tương ứng dataset person-only đã kiểm tra; không dùng nguyên protocol này cho MOT17 có distractor/ignore class mà chưa bổ sung preprocessing chính thức.

## 8. Tạm dừng và chạy tiếp

```powershell
# Dừng cache sau block đang chạy:
New-Item cache/val/STOP -ItemType File
# Khi tiếp tục:
Remove-Item cache/val/STOP
# Chạy lại nguyên lệnh cache, giữ nguyên output.

# Dừng tune sau trial đang chạy:
New-Item runs/tune_local/STOP -ItemType File
# Gỡ STOP và chạy lại nguyên lệnh tune để tiếp tục.
```

Trong replay `validate/evaluate`, đặt STOP ở gốc run tương ứng; mỗi worker lưu trạng thái sau block rồi dừng. Ctrl+C/đóng tiến trình cũng chạy tiếp được từ block/trial đã commit; phần chưa commit có thể cần làm lại. Không cần xóa output hoặc chạy lại từ đầu. Đừng mở hai writer cho cùng output; CLI có kiểm tra PID/create time.

Không thay model, split, ảnh, code, search hoặc GT giữa một run. Thay GT phải tạo manifest/run mới. Không đọc file pickle/checkpoint cache không tin cậy.

## 9. Khi GT được sửa nhưng ảnh vẫn giữ nguyên

Detection cache độc lập GT, nên có thể dùng lại cache và tạo manifest mới. Nếu đã có MOT predictions cho chính timeline đó, **chỉ chấm lại**, không inference/tracking lại:

```powershell
python -m exsort prepare --data "$Data" --split configs/split_seed20261024.json --role test --output runs/manifests/test_gt_v2.json
python -m exsort rescore --manifest runs/manifests/test_gt_v2.json --original-manifest runs/manifests/test.json --predictions runs/test_frozen/trackers/exsort/data --output runs/rescore_gt_v2 --workers 8
```

Hai manifest phải có image/timeline signature giống nhau. Nếu ảnh/frame đã bị thêm, xóa hoặc thay, không coi predictions cũ là còn hợp lệ.

## Cấu hình và kết quả tham chiếu

`configs/selected.json` là cấu hình mới nhất đã chọn trong vòng nghiên cứu cũ. [METHOD.md](docs/METHOD.md) mô tả rõ từng cơ chế và toàn bộ tham số; không gọi phiên bản này là “chỉ 7 tham số”.

**Các điểm dưới đây thuộc GT snapshot của lượt chạy 28–29/10/2026, trước lần sửa annotation 31/10–01/11. Chúng không phải kết quả trên GT hiện tại.**

| Method, test cũ | IDSW | HOTA↑ | DetA↑ | AssA↑ | IDF1↑ | MOTA↑ | FP↓ | FN↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TrackTrack full ReID |131|53,8074|60,5060|49,4444|63,1429|78,6141|4.743|19.427|
| exsort mới nhất, không ReID |39|57,0077|59,7425|55,5639|71,7511|76,8373|7.066|19.215|

exsort tốt hơn HOTA/IDF1/IDSW ở lượt này, nhưng FP nhiều hơn và MOTA thấp hơn TrackTrack. Hai pipeline có NMS/postprocessing khác nhau, nên không quy toàn bộ chênh lệch cho association hoặc ReID. Test đã từng được xem trong quá trình nghiên cứu trước đó; không mô tả đây là một test hoàn toàn chưa từng tiếp xúc. Lưu ý thêm sự chi phối của sequence morongmo trong pooled val và tương quan cùng cảnh; xem [TUNING.md](docs/TUNING.md).

## Cấu trúc repo và nguồn tham khảo

```text
exsort/core.py                    # thuật toán mới nhất; đổi import để portable
exsort/legacy.py                  # primitive/glare/recovery kế thừa
exsort/vendor/ocsort/             # OC-SORT Kalman/ORU và association
exsort/cache.py                   # inference và cache/import/verify
exsort/runner.py                  # replay/checkpoint/freeze/test/rescore
exsort/tuning.py                  # SMAC và ablation trên val
configs/                         # selected, baseline, search, split
docs/                            # method, tuning, legacy cache
history/                         # hash gốc và số liệu lịch sử, không chứa dữ liệu ảnh
tests/                           # kiểm tra tích hợp nhỏ
```

Phát triển từ các thành phần và ý tưởng của [OC-SORT](https://github.com/noahcao/OC_SORT), [TrackTrack](https://github.com/kamkyu94/TrackTrack), [ByteTrack](https://github.com/ifzhang/ByteTrack), [FilterPy](https://github.com/rlabbe/filterpy), [SMAC3](https://github.com/automl/SMAC3) và [TrackEval](https://github.com/JonathonLuiten/TrackEval). Giữ attribution/license các thành phần vendored ở [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Không bao gồm dataset, weight, ReID model, token, log cá nhân hoặc các script di chuyển/xóa dữ liệu của máy nghiên cứu.

Phạm vi đã kiểm tra và khác biệt số học khi inference mới được ghi tại [VERIFICATION.md](docs/VERIFICATION.md).
