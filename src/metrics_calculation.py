import json
from typing import List, Dict, Any

def calculate_iou(boxA: Dict[str, int], boxB: Dict[str, int]) -> float:
    x_left = max(boxA['x1'], boxB['x1'])
    y_top = max(boxA['y1'], boxB['y1'])
    x_right = min(boxA['x2'], boxB['x2'])
    y_bottom = min(boxA['y2'], boxB['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    areaA = (boxA['x2'] - boxA['x1']) * (boxA['y2'] - boxA['y1'])
    areaB = (boxB['x2'] - boxB['x1']) * (boxB['y2'] - boxB['y1'])
    union_area = areaA + areaB - intersection_area
    if union_area == 0:
        return 0.0
    return intersection_area / union_area


def calculate_st_iou(gt_boxes_list: List[Dict[str, Any]], pred_boxes_list: List[Dict[str, Any]]) -> float:
    gt_boxes = {b['frame']: {k: v for k, v in b.items() if k != 'frame'} for b in gt_boxes_list}
    pred_boxes = {b['frame']: {k: v for k, v in b.items() if k != 'frame'} for b in pred_boxes_list}

    gt_frames = set(gt_boxes.keys())
    pred_frames = set(pred_boxes.keys())

    intersection_frames = gt_frames.intersection(pred_frames)
    union_frames = gt_frames.union(pred_frames)

    if len(union_frames) == 0:
        return 0.0

    sum_iou_intersection = 0.0
    for frame in intersection_frames:
        sum_iou_intersection += calculate_iou(gt_boxes[frame], pred_boxes[frame])

    st_iou = sum_iou_intersection / len(union_frames)
    return st_iou


def load_data(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r') as f:
        return json.load(f)


ground_truth_data = load_data("/home/24kien.dhc/AeroEyes/AeroEyes/Data/Raw/train/annotations/annotations.json")
predicted_data = load_data("/home/24kien.dhc/AeroEyes/AeroEyes/src/runs/detect/final_predictions.json")

gt_dict = {item["video_id"]: item for item in ground_truth_data}
pred_dict = {item["video_id"]: item for item in predicted_data}

common_videos = sorted(list(set(gt_dict.keys()) & set(pred_dict.keys())))

# for sub in common_videos:
#     print(sub)

# print(len(gt_dict["Person1_0"]["annotations"][0]["bboxes"]))
# print()
# print(len(pred_dict["Person1_0"]["annotations"][0]["bboxes"]))

video_scores = []

for vid in common_videos:
    gt_annos = gt_dict[vid]["annotations"][0]["bboxes"]
    pred_annos = pred_dict[vid]["detections"][0]["bboxes"]

    st_iou = calculate_st_iou(gt_annos, pred_annos)
    video_scores.append(st_iou)
    print(f"{vid}: ST-IoU = {st_iou:.4f}")

if len(video_scores) > 0:
    final_score = sum(video_scores) / len(video_scores)
    print(f"\nFinal Score (mean ST-IoU across {len(video_scores)} videos): {final_score:.4f}")
else:
    print("No common videos found between prediction and ground-truth files.")
