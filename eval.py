"""
Evaluation script for Mask R-CNN lesion detection model.
Calculates precision, recall, F1-score, and mAP.
"""

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import cv2
import json
from tqdm import tqdm
from pathlib import Path

# Set up paths for Mask R-CNN
MASK_RCNN_DIR = os.path.join(os.getcwd(), 'Mask_RCNN')
sys.path.append(MASK_RCNN_DIR)

# Import Mask R-CNN modules
from mrcnn.config import Config
from mrcnn import model as modellib, utils, visualize

# Import the config and dataset classes from training script
from train_maskrcnn import LesionConfig, LesionDataset

class InferenceConfig(LesionConfig):
    """Configuration for inference"""
    # Set batch size to 1 for inference
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Skip detections with < 50% confidence so we can calculate precision/recall properly
    DETECTION_MIN_CONFIDENCE = 0.5


def compute_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes.
    Boxes are in format [y1, x1, y2, x2]
    """
    # Calculate intersection area
    y1_max = max(box1[0], box2[0])
    x1_max = max(box1[1], box2[1])
    y2_min = min(box1[2], box2[2])
    x2_min = min(box1[3], box2[3])

    intersection_area = max(0, y2_min - y1_max) * max(0, x2_min - x1_max)

    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    # Return IoU
    if union_area == 0:
        return 0
    return intersection_area / union_area


def compute_ap(precision, recall):
    """
    Compute Average Precision using the 11-point interpolation
    """
    # 11 point interpolation
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11
    return ap


def evaluate_model(model, dataset, config, save_dir=None, visualize_results=False):
    """
    Evaluate the model on the dataset.
    Returns precision, recall, F1 score and mAP.
    """
    # Create save directory if needed
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Initialize metrics
    APs = []
    precisions_at_iou = []
    recalls_at_iou = []
    f1_scores_at_iou = []

    # Confusion matrix
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # IoU threshold
    IOU_THRESHOLD = 0.5

    # Loop through all images in the dataset
    image_ids = dataset.image_ids

    for image_id in tqdm(image_ids):
        # Load image and ground truth data
        image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
            modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)

        # Get image details
        image_path = dataset.image_info[image_id]["path"]
        image_name = os.path.basename(image_path)

        # Run object detection
        results = model.detect([image], verbose=0)[0]

        # Extract results
        r_class_ids = results['class_ids']
        r_scores = results['scores']
        r_masks = results['masks']
        r_boxes = results['rois']

        # Compute IoU for each detected object vs each ground truth object
        iou_matrix = np.zeros((len(r_boxes), len(gt_boxes)))
        for i in range(len(r_boxes)):
            for j in range(len(gt_boxes)):
                iou_matrix[i, j] = compute_iou(r_boxes[i], gt_boxes[j])

        # Initialize detection matches
        detection_matches = []

        # For each ground truth, find the best matching detection
        for j in range(len(gt_boxes)):
            # Find max IoU
            if len(r_boxes) > 0:
                max_iou_idx = np.argmax(iou_matrix[:, j])
                max_iou = iou_matrix[max_iou_idx, j]

                # If IoU exceeds threshold, count as a match
                if max_iou >= IOU_THRESHOLD:
                    detection_matches.append(max_iou_idx)
                    true_positives += 1
                else:
                    false_negatives += 1
            else:
                false_negatives += 1

        # Count detections that didn't match any ground truth as false positives
        false_positives += len(r_boxes) - len(set(detection_matches))

        # Calculate precision-recall curve for this image
        # Sort detections by score
        indices = np.argsort(r_scores)[::-1]

        # Initialize precision and recall lists
        precisions = []
        recalls = []

        # Number of ground truth objects
        num_gt = len(gt_boxes)
        if num_gt == 0:
            # Skip images with no ground truth
            continue

        # Calculate precision and recall at each detection
        tp = 0
        fp = 0

        for i in indices:
            # Find best matching ground truth for this detection
            max_iou = 0
            match_idx = -1
            for j in range(len(gt_boxes)):
                iou = iou_matrix[i, j]
                if iou > max_iou:
                    max_iou = iou
                    match_idx = j

            # If IoU exceeds threshold, count as true positive
            if max_iou >= IOU_THRESHOLD:
                tp += 1
            else:
                fp += 1

            # Calculate precision and recall
            precisions.append(tp / (tp + fp))
            recalls.append(tp / num_gt)

        # Append precisions and recalls if we have detections
        if len(precisions) > 0:
            # Calculate AP for this image
            ap = compute_ap(np.array(precisions), np.array(recalls))
            APs.append(ap)

        # Visualize results if requested
        if visualize_results and save_dir:
            # Original image with ground truth
            visualize.display_instances(
                image,
                gt_boxes,
                gt_masks,
                gt_class_ids,
                dataset.class_names,
                title="Ground Truth",
                show_bbox=True,
                show_mask=True,
                figsize=(12, 12)
            )
            plt.savefig(os.path.join(save_dir, f"{image_name}_gt.png"))
            plt.close()

            # Original image with predictions
            visualize.display_instances(
                image,
                r_boxes,
                r_masks,
                r_class_ids,
                dataset.class_names,
                scores=r_scores,
                title="Predictions",
                show_bbox=True,
                show_mask=True,
                figsize=(12, 12)
            )
            plt.savefig(os.path.join(save_dir, f"{image_name}_pred.png"))
            plt.close()

    # Calculate final metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    mAP = np.mean(APs) if len(APs) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "mAP": mAP,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }


def main(args):
    # Create inference config
    inference_config = InferenceConfig()

    # Create model and load weights
    model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=args.model_dir)

    # Load weights
    print(f"Loading weights from {args.weights}")
    model.load_weights(args.weights, by_name=True)

    # Load dataset
    dataset = LesionDataset()
    dataset.load_lesions(args.dataset, args.subset)
    dataset.prepare()

    print(f"Running evaluation on {args.subset} set...")
    results = evaluate_model(
        model,
        dataset,
        inference_config,
        save_dir=args.output_dir,
        visualize_results=args.visualize
    )

    # Print results
    print("\nEvaluation Results:")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"mAP: {results['mAP']:.4f}")
    print(f"True Positives: {results['true_positives']}")
    print(f"False Positives: {results['false_positives']}")
    print(f"False Negatives: {results['false_negatives']}")

    # Save results to file
    if args.output_dir:
        results_file = os.path.join(args.output_dir, f"evaluation_results_{args.subset}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate Mask R-CNN for lesion detection')
    parser.add_argument('--dataset', required=True, help='Directory of the dataset')
    parser.add_argument('--subset', required=True, choices=['train', 'val', 'test'],
                        help='Which subset to evaluate on')
    parser.add_argument('--weights', required=True, help='Path to weights .h5 file')
    parser.add_argument('--model_dir', required=True, help='Path to logs and model files')
    parser.add_argument('--output_dir', help='Directory to save results and visualizations')
    parser.add_argument('--visualize', action='store_true', help='Visualize detection results')

    args = parser.parse_args()

    main(args)
