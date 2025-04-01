#!/bin/bash
#SBATCH --job-name=MedSAM2_Inference
#SBATCH --output=MedSAM2_Inference_%j.log
#SBATCH --error=MedSAM2_Inference_%j.log
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=main

###############################################################################
# MedSAM2 Inference Script for HPC
# This script runs inference using MedSAM2 on Mpox lesion images.
###############################################################################

# 1) Print start time and node info
echo "=========================================================="
echo "Starting MedSAM2 Inference on HPC"
echo "Current time: $(date)"
echo "Running on node: $(hostname)"
echo "GPU information:"
nvidia-smi
echo "Current directory: $(pwd)"
echo "=========================================================="

# 2) Define directories and settings
BASE_DIR=${SCRATCH:-$HOME}/MedSAM2_Mpox
MPOX_DATA_DIR=${BASE_DIR}/mpox_data
SCRIPTS_DIR=${BASE_DIR}/scripts
CHECKPOINT_DIR=${BASE_DIR}/checkpoints
RESULTS_DIR=${BASE_DIR}/results
ENV_SCRIPT=${BASE_DIR}/activate_env.sh

# 3) Activate MedSAM2 environment
echo "Activating MedSAM2 environment..."
source ${ENV_SCRIPT}

# 4) Create scripts directory and copy inference script
mkdir -p ${SCRIPTS_DIR}
cd ${SCRIPTS_DIR}

# Make utility functions available
source ${BASE_DIR}/utils.sh

# Create the inference script
cat > ${SCRIPTS_DIR}/run_medsam2_inference.py << 'EOF'
#!/usr/bin/env python3
"""
MedSAM2 inference script for Mpox lesion segmentation.
This script runs inference using the pre-trained MedSAM2 model on Mpox lesion images.
"""

import os
import sys
import numpy as np
import argparse
from pathlib import Path
import cv2
import time
import json
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import torch.nn.functional as F

# Import MedSAM2 related modules from the official repository
try:
    from segment_anything_2 import build_sam2_model
    from segment_anything_2 import SamPredictor
except ImportError:
    print("Error: segment_anything_2 package not found.")
    print("Make sure you've installed the official MedSAM2 package from bowang-lab.")
    print("Run the setup_medsam2.py script first.")
    sys.exit(1)

def ensure_dir(directory):
    """Make sure directory exists."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory

def load_npz_file(npz_path):
    """Load data from npz file."""
    try:
        data = np.load(npz_path, allow_pickle=True)
        return data
    except Exception as e:
        print(f"Error loading {npz_path}: {e}")
        return None

def build_medsam2_model(sam2_checkpoint, medsam2_checkpoint=None, model_cfg=None, device="cuda"):
    """Build and load the MedSAM2 model."""
    print(f"Building SAM2 model from {sam2_checkpoint}")
    
    if model_cfg and model_cfg.endswith(".yaml"):
        # Parse model config from YAML if provided
        import yaml
        with open(model_cfg, 'r') as f:
            cfg = yaml.safe_load(f)
            backbone_name = cfg.get('backbone_name', 'hiera_base_plus')
    else:
        # Guess model type from checkpoint name
        if "tiny" in sam2_checkpoint:
            backbone_name = "hiera_tiny"
        elif "small" in sam2_checkpoint:
            backbone_name = "hiera_small"
        elif "base_plus" in sam2_checkpoint:
            backbone_name = "hiera_base_plus"
        elif "large" in sam2_checkpoint:
            backbone_name = "hiera_large"
        else:
            print("Could not determine model type from checkpoint name.")
            print("Using default: hiera_base_plus")
            backbone_name = "hiera_base_plus"
    
    # Build the SAM2 model
    sam2_model = build_sam2_model(
        checkpoint=sam2_checkpoint,
        backbone_name=backbone_name
    )
    
    # Load MedSAM2 weights if provided
    if medsam2_checkpoint:
        print(f"Loading MedSAM2 weights from {medsam2_checkpoint}")
        checkpoint = torch.load(medsam2_checkpoint, map_location=device)
        
        # Check if the checkpoint is a state dict or a full checkpoint
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            sam2_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            sam2_model.load_state_dict(checkpoint)
    
    # Move model to device
    sam2_model.to(device=device)
    sam2_model.eval()
    
    return sam2_model

def run_inference_with_points(predictor, image, points=None, labels=None):
    """Run MedSAM2 inference with point prompts."""
    # Set image
    predictor.set_image(image)
    
    # Generate automatic points if not provided
    if points is None or labels is None:
        # Use simple method to generate foreground and background points
        h, w = image.shape[:2]
        center_y, center_x = h // 2, w // 2
        
        # Generate foreground point near the center
        fg_point = np.array([[center_x, center_y]])
        fg_label = np.array([1])
        
        # Generate background points near the corners
        margin = min(h, w) // 10
        bg_points = np.array([
            [margin, margin],  # Top-left
            [w - margin, margin],  # Top-right
            [margin, h - margin],  # Bottom-left
            [w - margin, h - margin]  # Bottom-right
        ])
        bg_labels = np.array([0, 0, 0, 0])
        
        # Combine points and labels
        points = np.vstack([fg_point, bg_points])
        labels = np.hstack([fg_label, bg_labels])
    
    # Convert to tensor
    points_torch = torch.from_numpy(points).unsqueeze(0).to(predictor.device)
    labels_torch = torch.from_numpy(labels).unsqueeze(0).to(predictor.device)
    
    # Run prediction
    masks, scores, _ = predictor.predict(
        point_coords=points_torch,
        point_labels=labels_torch,
        multimask_output=True,
    )
    
    # Get best mask
    best_idx = torch.argmax(scores)
    mask = masks[best_idx].cpu().numpy()
    score = scores[best_idx].item()
    
    return mask, score

def run_inference_with_box(predictor, image, bbox=None, bbox_shift=10):
    """Run MedSAM2 inference with bounding box prompt."""
    # Set image
    predictor.set_image(image)
    
    # Generate automatic box if not provided
    if bbox is None:
        # Use simple thresholding to estimate box
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Apply bbox_shift to expand the box slightly
            x = max(0, x - bbox_shift)
            y = max(0, y - bbox_shift)
            w = min(image.shape[1] - x, w + 2 * bbox_shift)
            h = min(image.shape[0] - y, h + 2 * bbox_shift)
            
            bbox = np.array([x, y, x + w, y + h])
        else:
            # If no contours, use a default box in the center
            h, w = image.shape[:2]
            center_x, center_y = w // 2, h // 2
            box_size = min(h, w) // 3
            bbox = np.array([
                center_x - box_size, center_y - box_size,
                center_x + box_size, center_y + box_size
            ])
    
    # Convert to tensor
    bbox_torch = torch.tensor(bbox, device=predictor.device).unsqueeze(0)
    
    # Run prediction
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=bbox_torch,
        multimask_output=True,
    )
    
    # Get best mask
    best_idx = torch.argmax(scores)
    mask = masks[best_idx].cpu().numpy()
    score = scores[best_idx].item()
    
    return mask, score, bbox

def process_single_image(npz_path, predictor, prompt_method="box", bbox_shift=10):
    """Process a single image with MedSAM2."""
    # Load data
    data = load_npz_file(npz_path)
    if data is None:
        return None
    
    # Get image and other metadata
    image = data['image']
    filename = str(data['filename'])
    
    # Get original size for resizing back
    if 'orig_size' in data:
        orig_size = tuple(data['orig_size'])
    else:
        orig_size = image.shape[:2]
    
    # Run inference based on prompt method
    inference_start = time.time()
    
    if prompt_method == "box":
        # Use bounding box for prompting
        bbox = data['bbox'] if 'bbox' in data else None
        mask_pred, confidence, bbox_used = run_inference_with_box(
            predictor, image, bbox, bbox_shift
        )
    else:
        # Use points for prompting
        points = data['points'] if 'points' in data else None
        labels = data['labels'] if 'labels' in data else None
        mask_pred, confidence = run_inference_with_points(
            predictor, image, points, labels
        )
        bbox_used = None
    
    inference_time = time.time() - inference_start
    
    # Get ground truth mask if available for comparison
    if 'mask' in data:
        mask_gt = data['mask']
    else:
        mask_gt = None
    
    # Calculate metrics if ground truth is available
    if mask_gt is not None:
        # Calculate Dice score
        intersection = np.logical_and(mask_pred, mask_gt).sum()
        dice_score = (2.0 * intersection) / (mask_pred.sum() + mask_gt.sum() + 1e-6)
        
        # Calculate IoU (Jaccard)
        union = np.logical_or(mask_pred, mask_gt).sum()
        iou_score = intersection / (union + 1e-6)
    else:
        dice_score = None
        iou_score = None
    
    # Create result dictionary
    result = {
        'npz_path': str(npz_path),
        'filename': filename,
        'mask_pred': mask_pred,
        'mask_gt': mask_gt,
        'confidence': confidence,
        'inference_time': inference_time,
        'dice_score': dice_score,
        'iou_score': iou_score,
        'bbox_used': bbox_used,
        'orig_size': orig_size
    }
    
    return result

def save_results(result, output_dir, save_masks=True, save_visualization=True):
    """Save inference results."""
    # Extract filename without extension
    filename = Path(result['filename']).stem
    
    # Create output directories
    masks_dir = ensure_dir(os.path.join(output_dir, "masks"))
    vis_dir = ensure_dir(os.path.join(output_dir, "visualizations"))
    
    # Save predicted mask
    if save_masks:
        mask_pred_path = os.path.join(masks_dir, f"{filename}_pred.png")
        cv2.imwrite(mask_pred_path, result['mask_pred'].astype(np.uint8) * 255)
    
    # Create visualization
    if save_visualization:
        # Get original image
        npz_data = load_npz_file(result['npz_path'])
        image = npz_data['image']
        
        # Create figure
        plt.figure(figsize=(15, 5))
        
        # Plot original image
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')
        
        # Plot predicted mask
        plt.subplot(1, 3, 2)
        plt.imshow(result['mask_pred'], cmap='gray')
        plt.title(f"Prediction (Conf: {result['confidence']:.3f})")
        plt.axis('off')
        
        # Plot overlay
        plt.subplot(1, 3, 3)
        overlay = image.copy()
        mask_rgb = np.zeros_like(overlay)
        mask_rgb[:,:,0] = result['mask_pred'] * 255  # Red channel
        overlay = cv2.addWeighted(overlay, 0.7, mask_rgb, 0.3, 0)
        
        # Draw bounding box if available
        if result['bbox_used'] is not None:
            bbox = result['bbox_used']
            cv2.rectangle(overlay, 
                        (int(bbox[0]), int(bbox[1])), 
                        (int(bbox[2]), int(bbox[3])), 
                        (0, 255, 0), 2)
        
        plt.imshow(overlay)
        plt.title(f"Overlay (Time: {result['inference_time']:.3f}s)")
        plt.axis('off')
        
        # Add metrics if available
        if result['dice_score'] is not None:
            plt.suptitle(f"Dice: {result['dice_score']:.3f}, IoU: {result['iou_score']:.3f}")
        
        # Save figure
        plt.tight_layout()
        vis_path = os.path.join(vis_dir, f"{filename}_vis.png")
        plt.savefig(vis_path)
        plt.close()
    
    return masks_dir, vis_dir

def run_medsam2_inference(npz_dir, output_dir, sam2_checkpoint, medsam2_checkpoint=None, 
                         model_cfg=None, prompt_method="box", bbox_shift=10, 
                         num_workers=4, device="cuda", save_visualization=True):
    """Run MedSAM2 inference on all images in the npz directory."""
    # Find all npz files
    npz_files = list(Path(npz_dir).glob("*.npz"))
    
    if len(npz_files) == 0:
        print(f"No npz files found in {npz_dir}")
        return
    
    print(f"Found {len(npz_files)} images for inference.")
    
    # Create output directory
    ensure_dir(output_dir)
    
    # Build MedSAM2 model
    sam2_model = build_medsam2_model(
        sam2_checkpoint, 
        medsam2_checkpoint,
        model_cfg,
        device
    )
    
    # Create predictor
    predictor = SamPredictor(sam2_model)
    
    # Process each image
    results = []
    metrics = []
    
    for npz_path in tqdm(npz_files, desc="Running inference"):
        # Process the image
        result = process_single_image(
            npz_path, 
            predictor, 
            prompt_method, 
            bbox_shift
        )
        
        if result:
            # Save results
            masks_dir, vis_dir = save_results(
                result, 
                output_dir, 
                save_masks=True, 
                save_visualization=save_visualization
            )
            
            # Collect metrics
            metrics.append({
                'filename': result['filename'],
                'confidence': result['confidence'],
                'inference_time': result['inference_time'],
                'dice_score': result['dice_score'],
                'iou_score': result['iou_score'],
            })
    
    # Save metrics to JSON
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Calculate average metrics
    avg_metrics = {
        'avg_confidence': np.mean([m['confidence'] for m in metrics]),
        'avg_inference_time': np.mean([m['inference_time'] for m in metrics]),
    }
    
    # Only include dice and IoU if available
    dice_scores = [m['dice_score'] for m in metrics if m['dice_score'] is not None]
    iou_scores = [m['iou_score'] for m in metrics if m['iou_score'] is not None]
    
    if dice_scores:
        avg_metrics['avg_dice_score'] = np.mean(dice_scores)
    if iou_scores:
        avg_metrics['avg_iou_score'] = np.mean(iou_scores)
    
    # Save average metrics
    avg_metrics_path = os.path.join(output_dir, "avg_metrics.json")
    with open(avg_metrics_path, 'w') as f:
        json.dump(avg_metrics, f, indent=2)
    
    # Print summary
    print("\nInference completed. Summary:")
    print(f"Total images processed: {len(metrics)}")
    print(f"Average inference time: {avg_metrics['avg_inference_time']:.3f} seconds")
    print(f"Average confidence: {avg_metrics['avg_confidence']:.3f}")
    
    if 'avg_dice_score' in avg_metrics:
        print(f"Average Dice score: {avg_metrics['avg_dice_score']:.3f}")
    if 'avg_iou_score' in avg_metrics:
        print(f"Average IoU score: {avg_metrics['avg_iou_score']:.3f}")
    
    print(f"\nResults saved to {output_dir}")
    print(f"Masks saved to {masks_dir}")
    if save_visualization:
        print(f"Visualizations saved to {vis_dir}")
    print(f"Metrics saved to {metrics_path}")
    
    return metrics, avg_metrics

def main():
    parser = argparse.ArgumentParser(description="Run MedSAM2 inference on Mpox images")
    parser.add_argument("--npz_dir", required=True, help="Directory containing npz files")
    parser.add_argument("--output_dir", default="./mpox_results", help="Output directory for results")
    parser.add_argument("--sam2_checkpoint", required=True, help="Path to SAM2 checkpoint")
    parser.add_argument("--medsam2_checkpoint", default=None, help="Path to MedSAM2 checkpoint")
    parser.add_argument("--model_cfg", default=None, help="Path to model config YAML file")
    parser.add_argument("--prompt_method", choices=["box", "points"], default="box",
                        help="Method for prompting SAM (box or points)")
    parser.add_argument("--bbox_shift", type=int, default=10,
                        help="Pixels to expand bounding box by (for box method)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device to run inference on")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for parallel processing")
    parser.add_argument("--no_visualization", action="store_true",
                        help="Skip generating visualizations")
    
    args = parser.parse_args()
    
    # Check if CUDA is available if requested
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Using CPU instead.")
        args.device = "cpu"
    
    # Run inference
    metrics, avg_metrics = run_medsam2_inference(
        args.npz_dir,
        args.output_dir,
        args.sam2_checkpoint,
        args.medsam2_checkpoint,
        args.model_cfg,
        args.prompt_method,
        args.bbox_shift,
        args.num_workers,
        args.device,
        not args.no_visualization
    )
    
    # Save configuration for reference
    config = vars(args)
    config['timestamp'] = str(Path.ctime(Path.cwd()))
    config['avg_metrics'] = avg_metrics
    
    with open(os.path.join(args.output_dir, "inference_config.json"), 'w') as f:
        json.dump(config, f, indent=2)

if __name__ == "__main__":
    main()
EOF
chmod +x ${SCRIPTS_DIR}/run_medsam2_inference.py

# 5) Create results directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
INFERENCE_RESULTS_DIR=${RESULTS_DIR}/inference_${TIMESTAMP}
mkdir -p ${INFERENCE_RESULTS_DIR}

# 6) Check for preprocessed data
NPZ_DIR=${MPOX_DATA_DIR}/npz_inference

if [ ! -d "${NPZ_DIR}" ] || [ -z "$(ls -A ${NPZ_DIR} 2>/dev/null)" ]; then
    echo "ERROR: No preprocessed data found in ${NPZ_DIR}."
    echo "Please run the data preparation script first: sbatch hpc_medsam2_dataprep.sh"
    exit 1
fi

# 7) Run inference with MedSAM2
# Use model path from checkpoints
SAM2_CHECKPOINT=${CHECKPOINT_DIR}/sam2_hiera_base_plus.pt
MEDSAM2_CHECKPOINT=${CHECKPOINT_DIR}/MedSAM2_pretrain.pth

# Check if models exist
if [ ! -f "${SAM2_CHECKPOINT}" ]; then
    echo "ERROR: SAM2 checkpoint not found at ${SAM2_CHECKPOINT}."
    echo "Please run the setup script first: sbatch hpc_medsam2_setup.sh"
    exit 1
fi

if [ ! -f "${MEDSAM2_CHECKPOINT}" ]; then
    echo "WARNING: MedSAM2 pretrained weights not found at ${MEDSAM2_CHECKPOINT}."
    echo "Will use the base SAM2 model without medical fine-tuning."
    MEDSAM2_CHECKPOINT_PARAM=""
else
    MEDSAM2_CHECKPOINT_PARAM="--medsam2_checkpoint ${MEDSAM2_CHECKPOINT}"
fi

echo "=========================================================="
echo "Running MedSAM2 inference on Mpox images..."
echo "Using SAM2 checkpoint: ${SAM2_CHECKPOINT}"
if [ ! -z "${MEDSAM2_CHECKPOINT_PARAM}" ]; then
    echo "Using MedSAM2 checkpoint: ${MEDSAM2_CHECKPOINT}"
else
    echo "No MedSAM2 checkpoint provided. Using base SAM2 model."
fi
echo "=========================================================="

# Run inference script
python ${SCRIPTS_DIR}/run_medsam2_inference.py \
    --npz_dir ${NPZ_DIR} \
    --output_dir ${INFERENCE_RESULTS_DIR} \
    --sam2_checkpoint ${SAM2_CHECKPOINT} \
    ${MEDSAM2_CHECKPOINT_PARAM} \
    --prompt_method box \
    --bbox_shift 10 \
    --device cuda \
    --num_workers $(nproc)

# 8) Print summary and next steps
echo "=========================================================="
echo "INFERENCE COMPLETED"
echo "=========================================================="
echo "Results directory: ${INFERENCE_RESULTS_DIR}"
echo ""

# Check if there are masks for fine-tuning
if [ -d "${MPOX_DATA_DIR}/npy" ] && [ ! -z "$(ls -A ${MPOX_DATA_DIR}/npy 2>/dev/null)" ]; then
    echo "Data for fine-tuning is available."
    echo ""
    echo "Next steps:"
    echo "1. Fine-tune the model (optional): sbatch hpc_medsam2_finetune.sh"
    echo "2. Evaluate results manually by examining the visualizations in ${INFERENCE_RESULTS_DIR}/visualizations"
else
    echo "Data for fine-tuning is not available."
    echo ""
    echo "Next steps:"
    echo "1. Evaluate results manually by examining the visualizations in ${INFERENCE_RESULTS_DIR}/visualizations"
    echo "2. If you want to fine-tune the model, please run the data preparation script with masks: sbatch hpc_medsam2_dataprep.sh"
fi
echo "=========================================================="
