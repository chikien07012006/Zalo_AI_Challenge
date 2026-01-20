# Zalo AI Challenge – Few‑Shot Drone Object Tracking

## Tổng quan
- Bài toán: phát hiện & bám theo đối tượng trong video drone với dữ liệu gán nhãn hạn chế (few-shot).
- Tiếp cận: fine-tune nhiều biến thể YOLO (từ nhẹ `yolo11n` đến lớn `yolo12l`, và các checkpoint tự huấn luyện `continuing_dp*`) để cân bằng kích thước mô hình và tốc độ suy luận.
- Quy trình gồm tiền xử lý (cắt frame + chuyển bbox sang YOLO), huấn luyện, suy luận song song đa GPU, tính ST-IoU, và trực quan hóa kết quả.

## Cấu trúc chính
- `Data_Preprocessing/data_retrieve.py`: tách frame từ video, chuyển bbox sang định dạng YOLO, lưu `images/` và `labels/`.
- `src/fine_tune.py`: kịch bản huấn luyện/fine-tune Ultralytics YOLO với nhiều cấu hình, tập trung tối ưu kích thước & thời gian (batch, mosaic, lr, cos lr, weight decay, v.v.).
- `src/inference.py`: suy luận video → JSON, chia danh sách video cho 2 GPU (ví dụ 6 & 7) để tăng tốc.
- `src/metrics_calculation.py`: tính ST-IoU giữa dự đoán và ground truth.
- `src/visualization.py`: vẽ bbox GT/Predict lên video để kiểm tra định tính.
- `src/runs/detect/*`: log huấn luyện, trọng số (`best.pt`, `last.pt`), biểu đồ và các file dự đoán (`final_predictions*.json`).

## Thiết lập môi trường
1) Cài đặt Python 3.10+ và PyTorch có CUDA.
2) Cài Ultralytics và phụ thuộc:
   ```bash
   pip install ultralytics opencv-python torch torchvision
   ```
3) Chuẩn bị dữ liệu thô:
   - Cấu trúc mỗi video: `<video_id>/drone_video.mp4`
   - File nhãn gốc: `annotations.json`

## Tiền xử lý dữ liệu
Chỉnh đường dẫn trong `Data_Preprocessing/data_retrieve.py`, sau đó chạy:
```bash
python Data_Preprocessing/data_retrieve.py
```
Kết quả: `output_folder/images` và `output_folder/labels` ở định dạng YOLO.

## Huấn luyện / Fine-tune
- Chỉnh `model = YOLO(<checkpoint>)` và tham số train trong `src/fine_tune.py`.
- Ví dụ (đã dùng để cân bằng tốc độ/độ chính xác):
```python
results = model.train(
    data="/path/to/data.yaml",
    imgsz=640, batch=16,
    lr0=5e-4, lrf=0.05, cos_lr=True,
    epochs=100, momentum=0.94, weight_decay=1e-4,
    patience=0, name="continuing_dp", save=True
)
```
- Các biến thể đã thử: 
  - Nhẹ: `yolo11n`, `yolo12n` (ưu tiên tốc độ, kích thước nhỏ cho few-shot).
  - Trung bình/lớn: `yolo12l`, các checkpoint tự huấn luyện `continuing_dp3/4/5` để tối ưu mAP/ST-IoU.

## Suy luận (tối ưu thời gian)
- `src/inference.py` chia danh sách video làm 2 phần và chạy song song trên 2 GPU:
```bash
python src/inference.py
```
- Tham số chính: `model_path`, `video_folder`, `output_json`, `conf_threshold`. Có thể thay đổi số GPU hoặc cách chia nếu phần cứng khác.

## Đánh giá
- Tính ST-IoU giữa dự đoán và nhãn gốc:
```bash
python src/metrics_calculation.py
```
- Điểm trung bình in ra cuối cùng; dùng để so sánh nhanh giữa các checkpoint/mô hình kích thước khác nhau.

## Trực quan hóa
- Vẽ bbox dự đoán/GT lên video để kiểm tra lỗi lệch khung/frame:
```bash
python src/abc.py
```
Hoặc dùng `src/visualization.py` để xuất video kèm thông tin frame & số bbox.

## Ghi chú triển khai few-shot
- Đa mô hình: thử nhiều backbone kích thước khác nhau để tìm điểm cân bằng tốc độ ↔ chính xác, phù hợp ràng buộc thời gian suy luận.
- Tăng cường: điều chỉnh mosaic/scale/translate, lr schedule (cosine), và trọng số loss (box/cls/dfl) để tận dụng ít dữ liệu.
- Đa GPU: chia video và dùng `torch.multiprocessing` để rút ngắn thời gian suy luận hàng loạt.
- Theo dõi kết quả: lưu mọi checkpoint/bbiểu đồ vào `src/runs/detect/*` để so sánh nhanh giữa các lần fine-tune.

## Đường dẫn mẫu (thay bằng của bạn)
- Checkpoint: `src/runs/detect/continuing_dp5/weights/best.pt`
- Dữ liệu thô: `/Data/Raw/train` và `/Data/Raw/public_test`
- Dự đoán: `src/runs/detect/final_predictions.json` hoặc `final_predictions_dp*.json`
# Zalo_AI_Challenge
# Zalo_AI_Challenge
# Zalo_AI_Challenge
# Zalo_AI_Challenge
