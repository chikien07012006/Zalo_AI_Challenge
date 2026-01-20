import os
import json
import cv2
from pathlib import Path

def convert_bbox_to_yolo(x1, y1, x2, y2, img_width, img_height):
    x_center = (x2 + x1) / 2.0 / img_width
    y_center = (y2 + y1) / 2.0 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return x_center, y_center, width, height
    

def process_video_annotations(annotation_file, output_folder, class_id=0):
    """
    Xử lý annotations và cắt ảnh từ video
    
    Args:
        annotation_file: đường dẫn đến file annotation JSON
        video_folder: thư mục chứa video files
        output_folder: thư mục đầu ra cho ảnh và labels
        class_id: ID class cho YOLO format (mặc định là 0)
    """
    
    # Đọc file annotation
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    images_dir = os.path.join(output_folder, 'images')
    labels_dir = os.path.join(output_folder, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    for video_data in annotations:
        video_id = video_data['video_id']
        video_path = os.path.join("/home/24kien.dhc/AeroEyes/AeroEyes/Data/Raw/train/samples", video_id, "drone_video.mp4")
        
        
        if not os.path.exists(video_path):
            print(f"Video {video_path} không tồn tại")
            continue
        
        print(f"Đang xử lý video: {video_id}")
        
        # Mở video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Không thể mở video: {video_path}")
            continue
        
        # Lấy thông tin video
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video info: {img_width}x{img_height}, FPS: {fps}, Total frames: {total_frames}")
        
        for annotation in video_data['annotations']:
            bboxes = annotation.get('bboxes', [])
            
            for bbox in bboxes:
                frame_number = bbox['frame']
                x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                
                # Lấy ảnh từ frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if not ret:
                    print(f"Không thể đọc frame {frame_number} từ video {video_id}")
                    continue
                
                # Tạo tên file
                image_filename = f"{video_id}_frame_{frame_number:06d}.jpg"
                label_filename = f"{video_id}_frame_{frame_number:06d}.txt"
                
                image_path = os.path.join(images_dir, image_filename)
                label_path = os.path.join(labels_dir, label_filename)
                
                # Lưu ảnh
                cv2.imwrite(image_path, frame)
                
                # Chuyển đổi và lưu bounding box theo YOLO format
                x_center, y_center, width, height = convert_bbox_to_yolo(
                    x1, y1, x2, y2, img_width, img_height
                )
                
                # Ghi file label
                with open(label_path, 'w') as label_file:
                    label_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
                print(f"Đã xử lý frame {frame_number} - Ảnh: {image_filename}, Label: {label_filename}")
        
        # Giải phóng video
        cap.release()
    
    print("Hoàn thành xử lý tất cả video!")




annotation_file = "/home/24kien.dhc/AeroEyes/AeroEyes/Data/Raw/train/annotations/annotations.json"  # Thay bằng đường dẫn đến file annotation của bạn         # Thư mục chứa video files
output_folder = "/home/24kien.dhc/AeroEyes/AeroEyes/Data/Preprocessed"        # Thư mục đầu ra

process_video_annotations(annotation_file, output_folder)

