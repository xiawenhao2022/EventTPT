import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use('Agg')

def calculate_iou(box1, box2):
    x1 = box1[0] - box1[2] / 2
    y1 = box1[1] - box1[3] / 2
    x2 = box1[0] + box1[2] / 2
    y2 = box1[1] + box1[3] / 2
    
    x1_p = box2[0] - box2[2] / 2
    y1_p = box2[1] - box2[3] / 2
    x2_p = box2[0] + box2[2] / 2
    y2_p = box2[1] + box2[3] / 2
    
    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)
    
    inter_area = max(inter_x2 - inter_x1, 0) * max(inter_y2 - inter_y1, 0)
    union_area = (x2 - x1) * (y2 - y1) + (x2_p - x1_p) * (y2_p - y1_p) - inter_area
    
    return inter_area / union_area

# 评估给定IoU阈值下的成功率
def evaluate_tracking(gt_boxes, test_boxes, iou_threshold=0.5):
    num_frames = len(gt_boxes)
    success = [1 if calculate_iou(gt, pred) >= iou_threshold else 0 for gt, pred in zip(gt_boxes, test_boxes)]
    return np.mean(success)

# 读取边界框数据的函数
def read_boxes(file_path, delimiter=','):
    try:
        return np.loadtxt(file_path, delimiter=delimiter)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return np.array([])
    
def read_test_files(test_dir):
    return {os.path.splitext(f)[0]: f for f in os.listdir(test_dir) if f.endswith('.txt')}

def calculate_precision_and_success_rate(gt_boxes, test_boxes, threshold):
    if len(gt_boxes) == 0 or len(test_boxes) == 0:
        return 0, 0
    
    num_correct = sum(calculate_iou(gt, pred) >= threshold for gt, pred in zip(gt_boxes, test_boxes))
    precision = num_correct / len(gt_boxes) if len(gt_boxes) > 0 else 0
    success_rate = evaluate_tracking(gt_boxes, test_boxes, threshold)
    
    return precision, success_rate

def evaluate_tracking_sequence(test_dir, gt_base_dir, iou_thresholds):
    test_files = read_test_files(test_dir)

    all_precisions = {threshold: [] for threshold in iou_thresholds}
    all_success_rates = {threshold: [] for threshold in iou_thresholds}
    
    for seq_id, test_file in test_files.items():
        print(f"Processing sequence {seq_id}...")

        gt_dir = os.path.join(gt_base_dir, seq_id)  # 假设gt文件在data/data/xx目录下
        gt_file_path = os.path.join(gt_dir, 'ground_truth.txt')

        test_file_path = os.path.join(test_dir, test_file)

        gt_boxes = read_boxes(gt_file_path)
        test_boxes = read_boxes(test_file_path)
        
        if len(gt_boxes) == 0 or len(test_boxes) == 0:
            print(f"No boxes found for sequence {seq_id}, skipping.")
            continue
        
        for threshold in iou_thresholds:
            precision, success_rate = calculate_precision_and_success_rate(gt_boxes, test_boxes, threshold)
            all_precisions[threshold].append(precision)
            all_success_rates[threshold].append(success_rate)
            print(f"  IoU {threshold}: Sequence {seq_id} - Precision = {precision:.4f}, Success Rate = {success_rate:.4f}")

    overall_avg_precisions = [np.mean(values) for values in all_precisions.values()]
    overall_avg_success_rates = [np.mean(values) for values in all_success_rates.values()]

    return all_precisions, overall_avg_precisions, overall_avg_success_rates

test_dir_path = 'RGBE_workspace/results/VisEvent/deep_rgbe'
gt_base_dir_path = 'data/data'
iou_thresholds = [0.5, 0.75]

all_precisions, overall_avg_precisions, overall_avg_success_rates = evaluate_tracking_sequence(test_dir_path, gt_base_dir_path, iou_thresholds)

for threshold, precisions in all_precisions.items():
    print(f"\nResults for IoU {threshold}:")
    for seq_id, precision in enumerate(precisions):
        print(f"  Sequence {seq_id} - Precision = {precision:.4f}")

print("\nOverall Average Results:")
for i, threshold in enumerate(iou_thresholds):
    print(f"IoU {threshold}: Precision = {overall_avg_precisions[i]:.4f}, Success Rate = {overall_avg_success_rates[i]:.4f}")