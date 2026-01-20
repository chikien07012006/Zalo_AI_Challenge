# predict_videos.py
import json
import cv2
import os
from ultralytics import YOLO
from pathlib import Path
import torch
import torch.multiprocessing as mp
from functools import partial

def predict_single_video(args):
    video_path, video_id, model_path, conf_threshold, device = args
    torch.cuda.set_device(device)
    model = YOLO(model_path)
    model.model.to(device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "video_id": video_id,
            "detections": []
        }

    video_detections = []
    current_interval = {"bboxes": []}
    last_frame = -2
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf_threshold, verbose=False, device=device)

        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    if frame_count == last_frame + 1:
                        current_interval["bboxes"].append({
                            "frame": frame_count,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2
                        })
                    else:
                        if current_interval["bboxes"]:
                            video_detections.append(current_interval)
                        current_interval = {"bboxes": [{
                            "frame": frame_count,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2
                        }]}
                    last_frame = frame_count

        frame_count += 1

    cap.release()

    if current_interval["bboxes"]:
        video_detections.append(current_interval)

    print(f"  [GPU {device}] Completed: {video_id} - {len(video_detections)} intervals")
    return {
        "video_id": video_id,
        "detections": video_detections
    }

def predict_videos_to_json(model_path, video_folder, output_json="predictions.json", conf_threshold=0.25):
    video_files = []
    for sub in sorted(os.listdir(video_folder)):
        path = os.path.join(video_folder, sub, "drone_video.mp4")
        if os.path.exists(path):
            video_files.append([path, sub])
    
    print(f"Found {len(video_files)} videos to process")

    # === CHIA VIDEO CHO 2 GPU (6 và 7) ===
    mid = len(video_files) // 2
    video_chunk_1 = video_files[:mid]
    video_chunk_2 = video_files[mid:]

    args_1 = [(path, vid, model_path, conf_threshold, 6) for path, vid in video_chunk_1]
    args_2 = [(path, vid, model_path, conf_threshold, 7) for path, vid in video_chunk_2]

    # === CHẠY SONG SONG ===
    mp.set_start_method('spawn', force=True)  # cần cho CUDA
    with mp.Pool(processes=2) as pool:
        results_1 = pool.map(predict_single_video, args_1) if args_1 else []
        results_2 = pool.map(predict_single_video, args_2) if args_2 else []
    
    submission_data = results_1 + results_2

    # Sắp xếp lại theo video_id để đảm bảo thứ tự (nếu cần)
    submission_data.sort(key=lambda x: x["video_id"])

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(submission_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nPredictions saved to: {output_json}")
    print(f"Total videos processed: {len(submission_data)}")
    
    return submission_data



# === Thay đường dẫn model và folder video của bạn ở đây ===
# === Thay đường dẫn model và folder video của bạn ở đây ===
model_path = "/home/24kien.dhc/AeroEyes/AeroEyes/src/runs/detect/continuing_dp5/weights/best.pt"
video_folder = "/home/24kien.dhc/AeroEyes/AeroEyes/Data/Raw/public_test/samples"  
output_json = "/home/24kien.dhc/AeroEyes/AeroEyes/src/runs/detect/public_test_1.json"
conf_threshold = 0.75

# === CHỈ THÊM DÒNG NÀY VÀO CUỐI FILE ===
if __name__ == '__main__':
    predictions = predict_videos_to_json(
        model_path=model_path,
        video_folder=video_folder,
        output_json=output_json,
        conf_threshold=conf_threshold
    )