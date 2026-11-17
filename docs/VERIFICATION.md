# Kiểm tra bản phát hành

## Cấu hình 1/2 giây ngày 17/11/2026

Replay cấu hình chính thức bằng API trên cache test cho output MOT trùng từng dòng với ablation 1/2 giây đã chạy. Phép đối chiếu gồm cả 48 sequence, 118.632 frame và 103.443 box đầu ra. FPS đọc từ metadata khớp protocol từng sequence. Fixture CPU kiểm tra cả ngưỡng 1/2 giây, trường hợp mask quay lại sau mốc 1 giây và detection ghép lại được ID cũ sau khi dừng xuất box.

## Kiểm tra lúc đóng gói ngày 03/11/2026

Các kiểm tra thực hiện trước public:

- Bộ kiểm tra synthetic CPU: checkpoint sau block + đoạn ghi chưa commit, resume cho prediction file cùng SHA với chạy liên tục; tune SMAC nhỏ trên val, freeze, test, chạy lại không đổi prediction; từ chối dùng test làm tuning input và từ chối sửa seal/config; GT thay đổi bị phát hiện; mask tính từ pixel trắng/đen; recovery duy nhất/ambiguous; toàn bộ fields HOTA/CLEAR/Identity/Count khớp native file-based TrackEval trên fixture có ID switch và FP.
- Đối chiếu snapshot exsort cuối với package portable: **400 frame /1.042 output boxes**, ID và tọa độ normalized **giống hệt**. Cùng cấu hình được chọn, ngẫu nhiên high/low/missing observations và vùng glare tổng hợp. Chỉ đổi import trong core/legacy; dependency OC-SORT vendored giữ byte-for-byte.
- Preprocessing FP16 so trực tiếp với `ValTransform` của fork TrackTrack/ByteTrack trên ảnh thật: tensor **giống hệt**.
- Adapter YOLOX-X chạy GPU thực trên **4 frame** của `150_sequence_32`, ảnh gốc số129–132. Checkpoint epoch15, 800×1440, FP16, batch4, CPU decode workers2, RTX5070Ti. So với cache val lịch sử: offsets giống, masks giống, số detection 6/6, toàn bộ tọa độ box giống; confidence có một sai khác nhỏ, max absolute≈`0.0004863739`. Vì vậy không khẳng định inference mới bit-exact với mọi cache cũ. Kernel/backend/batch có thể ảnh hưởng số học; dùng chính cache cũ khi cần lặp lại input cũ tuyệt đối.

Đợt đóng gói ngày 03/11 kiểm tra tính hoạt động và parity trên fixture và mẫu ảnh nêu trên. Kết quả benchmark của đợt này dùng snapshot GT lịch sử trong README.

Lệnh kiểm tra lại:

```powershell
python -m unittest discover -s tests -v
python scripts/check_reference.py --snapshot-dir PATH_TO_ORIGINAL_SNAPSHOT --halo-dev PATH_TO_ORIGINAL_HALO_DEV
python scripts/check_yolox_transform.py --yolox-root "third_party/TrackTrack/1. YOLOX" --image PATH_TO_IMAGE
```

Hai script parity cuối cần dependency/ảnh riêng của người dùng, không được kèm trong repo public. CI chỉ chạy fixture synthetic CPU.
