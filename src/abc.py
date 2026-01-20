# predict_videos.py
import json
import cv2
import os
from ultralytics import YOLO
from pathlib import Path

def predict_videos_to_json(model_path, video_folder, output_json="predictions.json", conf_threshold=0.25):
    model = YOLO(model_path)
    
    video_files = []
    
    for sub in sorted(os.listdir(video_folder)):
        print(sub)
        path = os.path.join(video_folder, sub, "drone_video.mp4")
        print(path)
        video_files.append([path, sub])
    
    print(f"Found {len(video_files)} videos to process")
    
    


model_path = "/home/24kien.dhc/AeroEyes/AeroEyes/src/runs/detect/train2/weights/best.pt"  # hoặc yolov8n.pt


video_folder = "/home/24kien.dhc/AeroEyes/AeroEyes/Data/Raw/train/samples"  
#video_folder = "/home/24kien.dhc/AeroEyes/AeroEyes/Data/Raw/public_test/samples"  


predictions = predict_videos_to_json(
    model_path=model_path,
    video_folder=video_folder,
    output_json="/home/24kien.dhc/AeroEyes/AeroEyes/src/runs/detect/final_predictions.json",
    conf_threshold=0.25
)