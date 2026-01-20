# visualize_submission.py
import json
import cv2
import os
from pathlib import Path

def visualize_submission_on_videos(submission_file, video_folder, output_folder="visualized_videos"):
    """
    Visualize kết quả từ file submission lên video gốc
    
    Args:
        submission_file: đường dẫn đến file submission.json
        video_folder: thư mục chứa video gốc
        output_folder: thư mục lưu video đã visualize
    """
    
    # Đọc file submission
    with open(submission_file, 'r') as f:
        submission_data = json.load(f)
    
    # Tạo thư mục output
    os.makedirs(output_folder, exist_ok=True)
    
    for video_data in submission_data:
        video_id = video_data['video_id']
        detections = video_data['detections']
        
        # Tìm video gốc
        video_path = find_video_file(video_folder, video_id)
        if not video_path:
            print(f"❌ Không tìm thấy video: {video_id}")
            continue
        
        print(f"🎥 Processing: {video_id}")
        
        # Visualize video
        output_path = os.path.join(output_folder, f"{video_id}_visualized.mp4")
        visualize_single_video(video_path, detections, output_path)
        
        print(f"✅ Saved: {output_path}")

def find_video_file(video_folder, video_id):
    """Tìm file video với các extension khác nhau"""
    extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV']
    
    for ext in extensions:
        video_path = os.path.join(video_folder, f"{video_id}{ext}")
        if os.path.exists(video_path):
            return video_path
        
        # Thử tìm không cần extension
        video_path_no_ext = os.path.join(video_folder, video_id)
        if os.path.exists(video_path_no_ext):
            return video_path_no_ext
    
    return None

def visualize_single_video(video_path, detections, output_path):
    """Visualize detections lên một video"""
    
    # Mở video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"   ERROR: Cannot open video {video_path}")
        return
    
    # Lấy thông tin video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Tạo video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"   Video info: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
    print(f"   Total detection groups: {len(detections)}")
    
    # Tạo lookup dictionary cho detections (theo frame number)
    detection_dict = {}
    for detection_group in detections:
        for bbox in detection_group['bboxes']:
            frame_num = bbox['frame']
            if frame_num not in detection_dict:
                detection_dict[frame_num] = []
            detection_dict[frame_num].append(bbox)
    
    frame_count = 0
    detected_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Kiểm tra nếu frame hiện tại có detection
        if frame_count in detection_dict:
            detected_frames += 1
            bboxes = detection_dict[frame_count]
            
            # Vẽ tất cả bounding boxes
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                
                # Vẽ bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Vẽ label
                label = f"Frame: {frame_count}"
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Vẽ thông tin frame
        info_text = f"Frame: {frame_count}/{total_frames} | Detections: {len(detection_dict.get(frame_count, []))}"
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Ghi frame
        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    
    print(f"   Processed: {frame_count} frames, Detected: {detected_frames} frames")

# Chạy visualize
if __name__ == "__main__":
    submission_file = "submission.json"  # file submission của bạn
    video_folder = "test_videos"         # thư mục video gốc
    output_folder = "visualized_results"
    
    visualize_submission_on_videos(submission_file, video_folder, output_folder)