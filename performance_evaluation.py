"""
   This script computes the performance metrics of an object detector model.
   The metrics considered as of now includes precision, and recall at [0, 100] IoU
   thresholds.
"""
import os
import csv
import numpy as np
from shapely.geometry import box

def compute_metrics_module(bboxes_pr, scores_pr, bboxes_gt, classes_gt, names, label_map, csv_name, step, run):
    true_pos_general = np.zeros((100, 1))
    false_pos_general = 0
    false_neg_general = 0
    true_neg_general = 0
    iouAll = []
    csv_name = run+csv_name
    write_label_map = not os.path.exists(csv_name)

    with open(csv_name, "a") as csv_file:
        csv_writer = csv.writer(csv_file)
        if write_label_map:
            csv_writer.writerow(label_map)
        for i, class_gt in enumerate(classes_gt):
            class_pr = np.argmax(scores_pr[i])
            print(class_pr, class_gt)
            bbox_gt = bboxes_gt[i]
            bbox_pr = bboxes_pr[i]
            bbox_gt = box(bbox_gt[0], bbox_gt[1], bbox_gt[2], bbox_gt[3])
            bbox_pr = box(bbox_pr[0], bbox_pr[1], bbox_pr[2], bbox_pr[3])
            intersection_pred_gt = bbox_gt.intersection(bbox_pr)
            union_pred_gt = bbox_gt.union(bbox_pr)
            iou_pred_gt = intersection_pred_gt.area / union_pred_gt.area
            if class_gt > 0: 
                iouAll.append(iou_pred_gt)
            if class_gt > 0 and class_pr == class_gt:
                true_pos_general[:int(100*iou_pred_gt)] += 1
            elif class_gt == 0 and class_pr > 0:
                false_pos_general += 1
            elif class_gt > 0 and class_pr == 0:
                false_neg_general += 1
            elif class_gt == 0 and class_pr == 0:
                true_neg_general += 1

            results = [names[i], class_gt, class_pr, iou_pred_gt, step]
            csv_writer.writerow(results)

    precision = np.round(true_pos_general / (true_pos_general + false_pos_general), 2)
    recall = np.round(true_pos_general / (true_pos_general + false_neg_general), 2)
    print("TRUE POS 50/90", true_pos_general[50], true_pos_general[90])
    print("False POS 50/90", false_pos_general)
    print("False Neg 50/90", false_neg_general)
    print("TRUE Neg 50/90", true_neg_general)
    print("MEAN IOU", np.mean(iouAll))
    return precision, recall
