# Import cache nghiên cứu cũ, không chạy lại detector

Chỉ dùng nếu bạn có các cache dưới đây và biết chắc ảnh/timeline khớp manifest mới. Importer kiểm tra số frame/kích thước, SHA block/mask nếu protocol cũ có ghi; **cache cũ không có đầy đủ image hashes**, nên cờ `--trust-legacy-images` là xác nhận provenance của caller, không phải một kiểm tra tự động đã chứng minh mọi ảnh giống nhau.

GT không lấy từ pickle cũ để scoring: runner luôn đọc `mot/gt/gt.txt` theo SHA manifest mới. Vì vậy GT sửa mà ảnh giữ nguyên có thể dùng lại detections/masks. Đổi ảnh/timeline cần cache mới.

## Val: cache mm đã gộp

```text
legacy-val/
  meta.json
  mm/<seq>.pkl   # tuple (T,W,H,offsets,detections,old_gt,features)
  mm/<seq>.npy   # packed masks, shape T x 32768
```

```powershell
$WeightSHA = (Get-FileHash weights/yolox_x_glaremot2_half_epoch15_fp16.pth -Algorithm SHA256).Hash.ToLower()
python -m exsort import-cache --manifest runs/manifests/val.json --format val-mm --source "D:\halo-repro\cache_glaremot2_valtrack" --detector-sha256 $WeightSHA --output cache/val --trust-legacy-images
python -m exsort verify-cache --manifest runs/manifests/val.json --cache cache/val
```

SHA weight phải thuộc detector **thực sự tạo cache**, không gán tùy ý SHA của một model mới. Val cache tham chiếu được tạo bằng epoch15 cùng checkpoint test; builder lịch sử ghi đường dẫn model đó. SHA checkpoint tham chiếu: `ffdbd2bcdd6030abb7800b2bfa7687ef29f16f897b375cdf02c3562a68556e59`. `meta.json` cũ không ghi SHA checkpoint nên importer lưu provenance này là thông tin caller cung cấp.

Không nhận pickle từ nguồn không tin cậy: Python pickle có thể thực thi code. Importer không cần/không dùng features ReID dù tuple cũ chứa chúng; chỉ giữ detection và mask.

## Test: protocol + các block NPZ cũ

```powershell
python -m exsort import-cache --manifest runs/manifests/test.json --format test-blocks --source "D:\HALO-SORT\results\exsort_round2_test_20261029_r4\protocol.json" --mask-map "D:\HALO-SORT\results\exsort_smac_test_20261028\cache_path_resolution.json" --output cache/test --trust-legacy-images
python -m exsort verify-cache --manifest runs/manifests/test.json --cache cache/test
```

Protocol nguồn cần `sequences` với width/height/frames, `blocks` chứa path/start/end/sha256 và:

- `off65`, `det65` trong detection block; nhánh NMS0,65, cap100, floor0,001.
- Nếu `rebuild_image_cache=true`, masks nằm trong chính block đó.
- Nếu không, `image_chunks` trỏ các block mask thật, cũng có SHA. `--mask-map` tùy chọn ánh xạ `mapping[old_path]=new_path` nếu đã chuyển chỗ cache. Không tìm thấy hoặc hash sai thì dừng, không tự điền mask 0.

Các ví dụ đường dẫn ổ D ở đây chỉ để chỉ đúng **artifact nghiên cứu cũ**. Code public không hardcode các đường dẫn đó; người dùng máy khác thay bằng nơi họ đặt cache/protocol.

## Format cache public sau import

```text
cache/val|test/
  cache.json           # detector/cache provenance, image identity, saturation settings
  verified.json
  <seq>/
    index.json         # start/end và SHA256 của từng block
    00000000.npz       # offsets + detections[N,5] + packed masks[T,32768]
    ...
```

`start/end` là offset 0-based, khoảng `[start,end)` trong phần sequence chọn; MOT output từ runner vẫn 1-based. Trong val, cache offset0 ứng với ảnh thứ `N//2` của sequence gốc. Không trộn offset này với tên file ảnh hoặc frame GT gốc.

Importer chỉ sao chép thông tin cần thiết vào format public, không xóa hay sửa cache gốc. Cache được nối tiếp sau interruption theo block. Metadata lưu legacy format/provenance để không nhầm cache import có audit ảnh đầy đủ.
