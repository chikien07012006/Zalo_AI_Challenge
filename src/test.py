import json
import cv2
import os

def visualize_video_with_predictions(
    video_root,
    video_id,
    pred_json,
    gt_json,
    output_path="visualized.mp4",
    thickness=2,
):
    # === Kiểm tra có GUI không (để tránh lỗi Qt xcb) ===
    can_show = os.environ.get("DISPLAY") is not None

    # Load JSON files
    with open(pred_json, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    with open(gt_json, "r", encoding="utf-8") as f:
        ground_truths = json.load(f)

    # Tìm annotation tương ứng video_id
    pred_data = next((item for item in predictions if item["video_id"] == video_id), None)
    gt_data = next((item for item in ground_truths if item["video_id"] == video_id), None)

    if pred_data is None:
        print(f"❌ Không tìm thấy {video_id} trong predictions.json")
        return
    if gt_data is None:
        print(f"❌ Không tìm thấy {video_id} trong annotations.json")
        return

    # Lấy đường dẫn video gốc
    video_path = os.path.join(video_root, video_id, "drone_video.mp4")
    if not os.path.exists(video_path):
        print(f"❌ Không tìm thấy video: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # Hàm chuyển annotations -> dict {frame: [bboxes]}
    def build_bbox_dict(annotations):
        bbox_dict = {}
        for interval in annotations:
            for b in interval["bboxes"]:
                frame_id = b["frame"]
                bbox_dict.setdefault(frame_id, []).append(b)
        return bbox_dict

    pred_boxes = build_bbox_dict(pred_data["annotations"])
    gt_boxes = build_bbox_dict(gt_data["annotations"])

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- Ground Truth (Xanh lá) ---
        if frame_idx in gt_boxes:
            for b in gt_boxes[frame_idx]:
                cv2.rectangle(frame, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (0, 255, 0), thickness)
                cv2.putText(frame, "GT", (b["x1"], b["y1"] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --- Predictions (Đỏ) ---
        if frame_idx in pred_boxes:
            for b in pred_boxes[frame_idx]:
                cv2.rectangle(frame, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (0, 0, 255), thickness)
                cv2.putText(frame, "Pred", (b["x1"], b["y2"] + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Ghi frame ra video
        out.write(frame)

        # Hiển thị nếu có GUI
        if can_show:
            cv2.imshow("Visualization", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    out.release()
    if can_show:
        cv2.destroyAllWindows()

    print(f"✅ Saved visualization video to: {output_path}")


# === Ví dụ chạy ===
video_root = "/home/24kien.dhc/AeroEyes/AeroEyes/Data/Raw/train/samples"
pred_json = "/home/24kien.dhc/AeroEyes/AeroEyes/src/runs/detect/final_predictions.json"
gt_json = "/home/24kien.dhc/AeroEyes/AeroEyes/Data/Raw/train/annotations/annotations.json"

visualize_video_with_predictions(
    video_root=video_root,
    video_id="Backpack_0",
    pred_json=pred_json,
    gt_json=gt_json,
    output_path="/home/24kien.dhc/AeroEyes/AeroEyes/src/runs/detect/vis_Backpack_0.mp4"
)
