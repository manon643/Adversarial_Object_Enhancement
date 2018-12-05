"""
   This script computes the performance metrics of an object detector model.
   The metrics considered as of now includes precision, and recall at [0, 100] IoU
   thresholds.
"""
import numpy as np
import pandas as pd
import json
import os
from itertools import chain
from shapely.geometry import box

def performance_curve(iou_all, number_boxes_gt, number_boxes_pr):
    """ This function computes the precision and recall curve given the iou scores throughout the dataset
        Input : iou_all - a lists of list storing iou scores for each image
                number_boxes_gt - number of ground truth boxes in the test data
                number_boxes_pr - number of predicted boxes in the test data

        Output : precision_curve - overall precision curve
                 recall_curve - overall recall curve
    """
    # Flatten the list of list
    iou_all = list(chain(*iou_all))

    # Initiate true positives array - threshold with 1 percent interval
    true_positives = np.zeros((100,1), dtype=np.float)
    for iou_img in iou_all:    # Go through the Images
        for iou_ind in iou_img: # Go through the predictions in images
            for threshold in range(0,100,1): # All Thresholds
                if iou_ind*100 > threshold:
                    true_positives[threshold] += 1

    # Normalize the true positive curve to find precision and recall curve
    overall_precision_curve = true_positives / (float(number_boxes_pr) + 1e-4)
    overall_recall_curve = true_positives / (float(number_boxes_gt) + 1e-4)

    # Convert to two digits precision
    for index in range(0,100,1):
         overall_precision_curve[index] = np.around(overall_precision_curve[index], 2)
         overall_recall_curve[index] = np.around(overall_recall_curve[index], 2)

    return [overall_precision_curve, overall_recall_curve]

def iou_scores(label_pr, bboxes_pr, scores_pr, bboxes_gt, classes_gt, confidence_threshold=0.5):
    """ This function computes the intersection over union between ground truth 
        and predicted bounding box for the image of interest
        Input : rect_gt - ground truth bounding boxes in an image in shapely format
                rect_pr - predicted bounding boxes in an images in shapely format

        Output : iou_all - a list storing intersection over unions between
                 ground truths and predictions on an image level
    """
    # Initiate lists to save the max IOUs and Indexes
    iou_all = []

    # Iterate over each ground truth and check predictions
    for index_gt in range(bboxes_gt.shape[0]):
        bbox_gt = bboxes_gt[index_gt]
        bbox_gt = box(bbox_gt[0], bbox_gt[1], bbox_gt[2], bbox_gt[3])

        # Predictions
        iou = []
        for index_pr in range(bboxes_pr.shape[0]):
            bbox_pr = bboxes_pr[index_pr]
            bbox_pr = box(bbox_pr[0], bbox_pr[1], bbox_pr[2], bbox_pr[3])

            # Compute the IOU w.r.t the closest truth
            if label_pr != classes_gt[index_gt] or scores_pr[index_pr] < confidence_threshold:
                continue

            intersection_pred_gt = bbox_gt.intersection(bbox_pr)
            union_pred_gt = bbox_gt.union(bbox_pr)
            iou_pred_gt = intersection_pred_gt.area / union_pred_gt.area

            # Append it to the image level list
            iou.append(iou_pred_gt)

        # Select the one that gives maximum IOU
        if not iou:
            iou_all.append(0)
        else:
            iou_all.append(round(max(iou),2))

    return iou_all

def performance_metric(bboxes_pr, scores_pr, bboxes_gt, classes_gt, confidence_threshold=0.5):
    """ Given the models and predictions on the test data this function assess
        the performances of the given models
    """
    # Initiate lists and counters
    precision_general = np.zeros((100,1))
    recall_general = np.zeros((100,1))

    # Go through each class label in ground truth
    number_boxes_gt_cl = 0
    for label_pr in bboxes_pr.keys():

        # Initialize class specific parameters
        iou_all = []
        number_boxes_pr = 0
        number_boxes_gt = 0
        # Iterate over each validation prediction by the model
        for index_img in range(bboxes_pr[label_pr].shape[0]):

            # Predicted Bounding Boxes - Initiate them, Also initiate a list to store
            # the IoUs for each ground truth
            iou_all.append([])
            iou_all[len(iou_all) - 1] = []

            # Compute the Precision and Recall Curves given the Predicted and Detected Bounding Boxes
            iou = iou_scores(label_pr, bboxes_pr[label_pr][index_img], scores_pr[label_pr][index_img], bboxes_gt[index_img], classes_gt[index_img])
            iou_all[len(iou_all) - 1].append(iou)

            # Needed for False Positives and False Negatives
            number_boxes_pr += sum(scores_pr[label_pr][index_img] > confidence_threshold)
            number_boxes_gt += sum(classes_gt[index_img] == label_pr)
     
        # Compute the overall recall and precision curve
        [precision_class, recall_class] = performance_curve(iou_all, number_boxes_gt, number_boxes_pr)

        # Update general precision and recall scores
        precision_general += (precision_class * float(number_boxes_gt))
        recall_general += (recall_class * float(number_boxes_gt))
        number_boxes_gt_cl += number_boxes_gt

    # Find the Mean Precision and Recall Across Different Classes
    precision_general = np.round(precision_general / float(number_boxes_gt_cl), 2)
    recall_general = np.round(recall_general / float(number_boxes_gt_cl), 2)
    
    return precision_general, recall_general

def compute_metrics(bboxes_pr, scores_pr, bboxes_gt, classes_gt):

    # Compute the precision and recall curves given the predictions and validations from a model
    precision, recall = performance_metric(bboxes_pr, scores_pr, bboxes_gt, classes_gt)

    return precision, recall


def compute_metrics_module(bboxes_pr, scores_pr, bboxes_gt, classes_gt):
    batch_size = scores_pr.shape[0]
    true_pos_general = np.zeros((100,1))
    recall_general = np.zeros((100,1))

    for i, class_obj in enumerate(classes_gt):
        class_pr = np.argmax(scores_pr[i])
        print(class_pr, class_obj)
        if class_pr!=class_obj or np.max(scores_pr[i])<0.3:
           continue
        bbox_gt = bboxes_gt[i]
        bbox_pr = bboxes_pr[i]
        bbox_gt = box(bbox_gt[0], bbox_gt[1], bbox_gt[2], bbox_gt[3])
        bbox_pr = box(bbox_pr[0], bbox_pr[1], bbox_pr[2], bbox_pr[3])
        intersection_pred_gt = bbox_gt.intersection(bbox_pr)
        union_pred_gt = bbox_gt.union(bbox_pr)
        iou_pred_gt = intersection_pred_gt.area / union_pred_gt.area
        true_pos_general[:int(100*iou_pred_gt)]+=1

    precision = np.round(true_pos_general / float(batch_size), 2)
    recall = np.round(true_pos_general / float(batch_size), 2)
    return precision, recall
