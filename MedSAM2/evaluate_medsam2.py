#!/usr/bin/env python3
"""
Evaluation script for MedSAM2 on Mpox lesion segmentation.
This script evaluates segmentation performance and compares different models.
"""

import os
import sys
import numpy as np
import argparse
from pathlib import Path
import cv2
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def ensure_dir(directory):
    """Make sure directory exists."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory

def load_image(image_path):
    """Load an image from file."""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error reading image: {image_path}")
            return None
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

def load_mask(mask_path):
    """Load a mask from file."""
    try:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Error reading mask: {mask_path}")
            return None
        return (mask > 0).astype(np.uint8)
    except Exception as e:
        print(f"Error loading {mask_path}: {e}")
        return None

def compute_metrics(pred_mask, gt_mask):
    """Compute evaluation metrics for a single image."""
    # Flatten masks
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()
    
    # Calculate confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(gt_flat, pred_flat, labels=[0, 1]).ravel()
    
    # Calculate metrics
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate Dice coefficient (equivalent to F1 score)
    dice = f1_score
    
    # Calculate Intersection over Union (IoU)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    
    # Calculate Hausdorff distance
    try:
        # Convert binary masks to points
        pred_points = np.argwhere(pred_mask > 0)
        gt_points = np.argwhere(gt_mask > 0)
        
        if len(pred_points) > 0 and len(gt_points) > 0:
            # Calculate Hausdorff distance
            from scipy.spatial.distance import directed_hausdorff
            hausdorff_1 = directed_hausdorff(pred_points, gt_points)[0]
            hausdorff_2 = directed_hausdorff(gt_points, pred_points)[0]
            hausdorff_dist = max(hausdorff_1, hausdorff_2)
        else:
            hausdorff_dist = np.nan
    except Exception as e:
        hausdorff_dist = np.nan
    
    # Calculate Boundary F1 score (adapted from MedSAM2 paper)
    try:
        # Get contours of masks
        pred_contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        gt_contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # Create boundary masks
        pred_boundary = np.zeros_like(pred_mask)
        gt_boundary = np.zeros_like(gt_mask)
        
        # Draw contours on boundary masks
        cv2.drawContours(pred_boundary, pred_contours, -1, 1, 1)
        cv2.drawContours(gt_boundary, gt_contours, -1, 1, 1)
        
        # Calculate boundary F1
        boundary_tp = np.sum(np.logical_and(pred_boundary, gt_boundary))
        boundary_fp = np.sum(np.logical_and(pred_boundary, np.logical_not(gt_boundary)))
        boundary_fn = np.sum(np.logical_and(np.logical_not(pred_boundary), gt_boundary))
        
        boundary_precision = boundary_tp / (boundary_tp + boundary_fp) if (boundary_tp + boundary_fp) > 0 else 0
        boundary_recall = boundary_tp / (boundary_tp + boundary_fn) if (boundary_tp + boundary_fn) > 0 else 0
        boundary_f1 = 2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall) if (boundary_precision + boundary_recall) > 0 else 0
    except Exception as e:
        boundary_f1 = np.nan
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'dice': dice,
        'iou': iou,
        'hausdorff': hausdorff_dist,
        'boundary_f1': boundary_f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }
    
    return metrics

def evaluate_model(pred_dir, gt_dir, output_dir, num_workers=4):
    """Evaluate a model by comparing predicted masks with ground truth."""
    # Find all predicted masks
    pred_masks = glob.glob(os.path.join(pred_dir, "*.png"))
    
    if len(pred_masks) == 0:
        print(f"No predicted masks found in {pred_dir}")
        return None
    
    # Create list of ground truth mask paths
    gt_masks = []
    for pred_path in pred_masks:
        pred_filename = os.path.basename(pred_path)
        # Remove suffix like "_pred" to get original filename
        original_filename = pred_filename.replace("_pred", "").replace("_mask", "")
        
        # Try different possible ground truth mask filenames
        gt_candidates = [
            os.path.join(gt_dir, original_filename),
            os.path.join(gt_dir, f"{Path(original_filename).stem}_mask.png"),
            os.path.join(gt_dir, f"{Path(original_filename).stem}_gt.png")
        ]
        
        gt_path = None
        for candidate in gt_candidates:
            if os.path.exists(candidate):
                gt_path = candidate
                break
        
        if gt_path:
            gt_masks.append(gt_path)
        else:
            print(f"Warning: No matching ground truth found for {pred_filename}")
            gt_masks.append(None)
    
    # Filter out pairs where ground truth is missing
    valid_pairs = [(pred, gt) for pred, gt in zip(pred_masks, gt_masks) if gt is not None]
    
    if len(valid_pairs) == 0:
        print("No valid image pairs found. Cannot perform evaluation.")
        return None
    
    print(f"Found {len(valid_pairs)} valid image pairs for evaluation.")
    
    # Process each image pair in parallel
    results = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for pred_path, gt_path in valid_pairs:
            futures.append(executor.submit(process_image_pair, pred_path, gt_path))
        
        for future in tqdm(futures, desc="Evaluating images"):
            result = future.result()
            if result:
                results.append(result)
    
    # Calculate average metrics
    avg_metrics = {}
    for metric in results[0]['metrics'].keys():
        valid_values = [r['metrics'][metric] for r in results if not np.isnan(r['metrics'][metric])]
        if valid_values:
            avg_metrics[metric] = np.mean(valid_values)
        else:
            avg_metrics[metric] = np.nan
    
    # Create evaluation report
    report = {
        'average_metrics': avg_metrics,
        'per_image_results': results
    }
    
    # Save report
    ensure_dir(output_dir)
    report_path = os.path.join(output_dir, "evaluation_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create visualizations
    create_evaluation_visualizations(results, output_dir)
    
    # Print summary
    print("\nEvaluation completed. Summary:")
    for metric, value in avg_metrics.items():
        if not np.isnan(value):
            print(f"{metric}: {value:.4f}")
    
    print(f"\nDetailed report saved to {report_path}")
    print(f"Visualizations saved to {output_dir}")
    
    return report

def process_image_pair(pred_path, gt_path):
    """Process a single pair of predicted and ground truth masks."""
    # Load masks
    pred_mask = load_mask(pred_path)
    gt_mask = load_mask(gt_path)
    
    if pred_mask is None or gt_mask is None:
        return None
    
    # Make sure masks have the same size
    if pred_mask.shape != gt_mask.shape:
        pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Compute metrics
    metrics = compute_metrics(pred_mask, gt_mask)
    
    return {
        'pred_path': pred_path,
        'gt_path': gt_path,
        'metrics': metrics
    }

def create_evaluation_visualizations(results, output_dir):
    """Create visualizations for evaluation results."""
    vis_dir = ensure_dir(os.path.join(output_dir, "visualizations"))
    
    # Extract metrics for plotting
    metrics_to_plot = ['dice', 'iou', 'precision', 'recall', 'boundary_f1']
    data = {metric: [r['metrics'][metric] for r in results] for metric in metrics_to_plot}
    
    # Create performance distribution plots
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(2, 3, i+1)
        sns.histplot(data[metric], kde=True)
        plt.xlabel(metric.capitalize())
        plt.ylabel('Frequency')
        plt.title(f'{metric.capitalize()} Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "metrics_distribution.png"))
    plt.close()
    
    # Create box plots
    plt.figure(figsize=(15, 6))
    df = pd.DataFrame(data)
    sns.boxplot(data=df)
    plt.title('Distribution of Evaluation Metrics')
    plt.savefig(os.path.join(vis_dir, "metrics_boxplot.png"))
    plt.close()
    
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Between Metrics')
    plt.savefig(os.path.join(vis_dir, "metrics_correlation.png"))
    plt.close()
    
    # Create comparison visualizations for best and worst cases
    create_comparison_visualizations(results, vis_dir)

def create_comparison_visualizations(results, vis_dir):
    """Create visualizations comparing predictions with ground truth."""
    # Sort results by different metrics
    metrics_to_sort = ['dice', 'iou', 'boundary_f1']
    
    for metric in metrics_to_sort:
        # Sort by metric (ascending)
        sorted_results = sorted(results, key=lambda x: x['metrics'][metric])
        
        # Get worst and best cases
        worst_cases = sorted_results[:min(5, len(sorted_results))]
        best_cases = sorted_results[-min(5, len(sorted_results)):]
        
        # Create directory for this metric
        metric_dir = ensure_dir(os.path.join(vis_dir, metric))
        
        # Visualize worst cases
        for i, case in enumerate(worst_cases):
            create_comparison_image(case, os.path.join(metric_dir, f"worst_{i+1}.png"), metric)
        
        # Visualize best cases
        for i, case in enumerate(best_cases):
            create_comparison_image(case, os.path.join(metric_dir, f"best_{i+1}.png"), metric)

def create_comparison_image(case, output_path, metric):
    """Create a comparison image between prediction and ground truth."""
    # Load ground truth and prediction
    gt_mask = load_mask(case['gt_path'])
    pred_mask = load_mask(case['pred_path'])
    
    if gt_mask is None or pred_mask is None:
        return
    
    # Load original image if possible
    original_filename = os.path.basename(case['gt_path']).replace("_mask", "").replace("_gt", "")
    original_dir = os.path.dirname(os.path.dirname(case['gt_path']))
    
    original_candidates = [
        os.path.join(original_dir, "images", original_filename),
        os.path.join(original_dir, "images", f"{Path(original_filename).stem}.jpg"),
        os.path.join(original_dir, "images", f"{Path(original_filename).stem}.png")
    ]
    
    original_image = None
    for candidate in original_candidates:
        if os.path.exists(candidate):
            original_image = load_image(candidate)
            break
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    # Plot original image if available
    if original_image is not None:
        plt.subplot(1, 4, 1)
        plt.imshow(original_image)
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.imshow(gt_mask, cmap='gray')
        plt.title("Ground Truth")
        plt.axis('off')
        
        plt.subplot(1, 4, 3)
        plt.imshow(pred_mask, cmap='gray')
        plt.title("Prediction")
        plt.axis('off')
        
        plt.subplot(1, 4, 4)
        # Create overlay showing TP, FP, FN
        overlay = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
        # True positive: green
        overlay[np.logical_and(gt_mask, pred_mask)] = [0, 255, 0]
        # False positive: red
        overlay[np.logical_and(np.logical_not(gt_mask), pred_mask)] = [255, 0, 0]
        # False negative: blue
        overlay[np.logical_and(gt_mask, np.logical_not(pred_mask))] = [0, 0, 255]
        
        plt.imshow(overlay)
        plt.title("Error Map")
        plt.axis('off')
    else:
        plt.subplot(1, 3, 1)
        plt.imshow(gt_mask, cmap='gray')
        plt.title("Ground Truth")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(pred_mask, cmap='gray')
        plt.title("Prediction")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        # Create overlay showing TP, FP, FN
        overlay = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
        # True positive: green
        overlay[np.logical_and(gt_mask, pred_mask)] = [0, 255, 0]
        # False positive: red
        overlay[np.logical_and(np.logical_not(gt_mask), pred_mask)] = [255, 0, 0]
        # False negative: blue
        overlay[np.logical_and(gt_mask, np.logical_not(pred_mask))] = [0, 0, 255]
        
        plt.imshow(overlay)
        plt.title("Error Map")
        plt.axis('off')
    
    # Add metric value to title
    metric_value = case['metrics'][metric]
    plt.suptitle(f"{metric.capitalize()}: {metric_value:.4f}")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def compare_models(model_results, output_dir):
    """Compare results from multiple models."""
    if len(model_results) < 2:
        print("Need at least two models to compare.")
        return
    
    # Create output directory
    compare_dir = ensure_dir(os.path.join(output_dir, "model_comparison"))
    
    # Extract model names and average metrics
    model_names = list(model_results.keys())
    metrics_to_compare = ['dice', 'iou', 'precision', 'recall', 'f1_score', 'boundary_f1']
    
    # Create comparison dataframe
    comparison_data = {
        'model': [],
        'metric': [],
        'value': []
    }
    
    for model_name, result in model_results.items():
        for metric in metrics_to_compare:
            value = result['average_metrics'].get(metric, np.nan)
            if not np.isnan(value):
                comparison_data['model'].append(model_name)
                comparison_data['metric'].append(metric)
                comparison_data['value'].append(value)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create grouped bar chart
    plt.figure(figsize=(15, 8))
    sns.barplot(x='metric', y='value', hue='model', data=comparison_df)
    plt.title('Model Comparison')
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(compare_dir, "model_comparison_bar.png"))
    plt.close()
    
    # Create radar chart for each model
    plt.figure(figsize=(10, 10))
    
    # Compute the angles for the radar chart
    num_metrics = len(metrics_to_compare)
    angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the circle
    
    # Create subplot with polar projection
    ax = plt.subplot(111, polar=True)
    
    # Draw the radar chart for each model
    for model_name, result in model_results.items():
        values = [result['average_metrics'].get(metric, 0) for metric in metrics_to_compare]
        values += values[:1]  # Close the circle
        
        ax.plot(angles, values, linewidth=2, label=model_name)
        ax.fill(angles, values, alpha=0.1)
    
    # Set the labels and styling
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_to_compare)
    ax.set_ylim(0, 1)
    plt.legend(loc='upper right')
    plt.title('Model Comparison Radar Chart')
    plt.tight_layout()
    plt.savefig(os.path.join(compare_dir, "model_comparison_radar.png"))
    plt.close()
    
    # Save comparison data
    comparison_df.to_csv(os.path.join(compare_dir, "model_comparison.csv"), index=False)
    
    # Create a summary table
    summary_table = []
    for metric in metrics_to_compare:
        row = {'metric': metric}
        for model_name in model_names:
            row[model_name] = model_results[model_name]['average_metrics'].get(metric, np.nan)
        summary_table.append(row)
    
    summary_df = pd.DataFrame(summary_table)
    summary_df.to_csv(os.path.join(compare_dir, "model_comparison_summary.csv"), index=False)
    
    # Print summary
    print("\nModel Comparison Summary:")
    print(summary_df.to_string(index=False))
    
    return summary_df

def main():
    parser = argparse.ArgumentParser(description="Evaluate MedSAM2 on Mpox lesion segmentation")
    parser.add_argument("--pred_dir", required=True, help="Directory containing predicted masks")
    parser.add_argument("--gt_dir", required=True, help="Directory containing ground truth masks")
    parser.add_argument("--output_dir", default="./evaluation_results", help="Output directory for evaluation results")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--compare", nargs='+', help="List of additional prediction directories to compare with")
    parser.add_argument("--model_names", nargs='+', help="Names for the models being compared")
    
    args = parser.parse_args()
    
    # Run evaluation for the main model
    report = evaluate_model(args.pred_dir, args.gt_dir, args.output_dir, args.num_workers)
    
    # Compare with additional models if requested
    if args.compare:
        if args.model_names and len(args.model_names) != len(args.compare) + 1:
            print("Warning: Number of model names does not match number of models. Using default names.")
            model_names = [f"Model {i+1}" for i in range(len(args.compare) + 1)]
        elif args.model_names:
            model_names = args.model_names
        else:
            model_names = [f"Model {i+1}" for i in range(len(args.compare) + 1)]
        
        # Collect results from all models
        model_results = {model_names[0]: report}
        
        for i, compare_dir in enumerate(args.compare):
            compare_output_dir = os.path.join(args.output_dir, f"compare_{i+1}")
            compare_report = evaluate_model(compare_dir, args.gt_dir, compare_output_dir, args.num_workers)
            if compare_report:
                model_results[model_names[i+1]] = compare_report
        
        # Compare models
        compare_models(model_results, args.output_dir)

if __name__ == "__main__":
    main()
