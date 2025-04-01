#!/usr/bin/env python3
"""
MedSAM2 inference script for Mpox lesion segmentation.
This script runs inference using the fine-tuned MedSAM2 model on Mpox lesion images.
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
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import torch.nn.functional as F

# Import MedSAM2 related modules
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("Error: sam2 package not found.")
    print("Make sure you've installed the SAM2 package.")
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

def load_image(image_path, target_size=(1024, 1024)):
    """Load and preprocess an image."""
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error reading image: {image_path}")
        return None
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize image (preserve aspect ratio)
    h, w = image.shape[:2]
    ratio = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * ratio), int(w * ratio)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create a black canvas of target size
    canvas = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    
    # Paste the resized image onto the canvas (center it)
    y_offset = (target_size[0] - new_h) // 2
    x_offset = (target_size[1] - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas, (h, w), (y_offset, x_offset, new_h, new_w)

def run_inference_with_box(predictor, image, bbox=None, bbox_shift=10):
    """Run inference with bounding box prompt."""
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
    
    # Convert to torch tensor
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

def process_single_image(image_data, predictor, prompt_method="box", bbox_shift=10):
    """Process a single image with MedSAM2."""
    # Case 1: We have a preprocessed npz file
    if isinstance(image_data, str) and image_data.endswith('.npz'):
        # Load data
        data = load_npz_file(image_data)
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
        
        # Get optional bbox if available
        bbox = data.get('bbox', None)
        
    # Case 2: We have a raw image file path
    elif isinstance(image_data, str) and os.path.isfile(image_data):
        # Load and preprocess image
        image, orig_size, crop_info = load_image(image_data)
        if image is None:
            return None
        
        filename = os.path.basename(image_data)
        bbox = None  # Auto-generate bbox
        
    # Case 3: We have a numpy array image
    elif isinstance(image_data, np.ndarray):
        image = image_data
        orig_size = image.shape[:2]
        filename = "image_array"
        bbox = None  # Auto-generate bbox
    
    else:
        print(f"Unsupported image data type: {type(image_data)}")
        return None
    
    # Run inference
    inference_start = time.time()
    mask_pred, confidence, bbox_used = run_inference_with_box(
        predictor, image, bbox, bbox_shift
    )
    inference_time = time.time() - inference_start
    
    # Create result dictionary
    result = {
        'image': image,
        'filename': filename,
        'mask_pred': mask_pred,
        'confidence': confidence,
        'inference_time': inference_time,
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
        # Get image
        image = result['image']
        
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
        
        # Save figure
        plt.tight_layout()
        vis_path = os.path.join(vis_dir, f"{filename}_vis.png")
        plt.savefig(vis_path)
        plt.close()
    
    return masks_dir, vis_dir

def run_inference(input_dir, output_dir, sam2_checkpoint, medsam2_checkpoint=None, 
                model_cfg=None, prompt_method="box", bbox_shift=10, 
                num_workers=4, device="cuda", save_visualization=True):
    """Run MedSAM2 inference on all images."""
    # Check if input is a directory or a single file
    if os.path.isdir(input_dir):
        # Find all image or npz files
        input_files = []
        for ext in ['.npz', '.jpg', '.jpeg', '.png', '.bmp']:
            input_files.extend(list(Path(input_dir).glob(f"**/*{ext}")))
        
        if len(input_files) == 0:
            print(f"No input files found in {input_dir}")
            return
        
        print(f"Found {len(input_files)} files for inference.")
    elif os.path.isfile(input_dir):
        input_files = [Path(input_dir)]
        print(f"Running inference on single file: {input_dir}")
    else:
        print(f"Input path {input_dir} is invalid")
        return
    
    # Create output directory
    ensure_dir(output_dir)
    
    # Build SAM2 model and create predictor
    print(f"Loading model from {sam2_checkpoint}")
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    
    # Load MedSAM2 weights if provided
    if medsam2_checkpoint and os.path.isfile(medsam2_checkpoint):
        print(f"Loading MedSAM2 weights from {medsam2_checkpoint}")
        checkpoint = torch.load(medsam2_checkpoint, map_location=device)
        
        # Check if checkpoint is a full state_dict or a training checkpoint
        if 'model' in checkpoint:
            # Training checkpoint format
            sam2_model.load_state_dict(checkpoint['model'])
        else:
            # Direct state_dict format
            sam2_model.load_state_dict(checkpoint)
    
    # Create predictor
    predictor = SAM2ImagePredictor(sam2_model)
    
    # Process each image
    results = []
    metrics = []
    
    for input_file in tqdm(input_files, desc="Running inference"):
        # Process the image
        result = process_single_image(
            str(input_file), 
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
    
    # Save average metrics
    avg_metrics_path = os.path.join(output_dir, "avg_metrics.json")
    with open(avg_metrics_path, 'w') as f:
        json.dump(avg_metrics, f, indent=2)
    
    # Print summary
    print("\nInference completed. Summary:")
    print(f"Total images processed: {len(metrics)}")
    print(f"Average inference time: {avg_metrics['avg_inference_time']:.3f} seconds")
    print(f"Average confidence: {avg_metrics['avg_confidence']:.3f}")
    
    print(f"\nResults saved to {output_dir}")
    print(f"Masks saved to {masks_dir}")
    if save_visualization:
        print(f"Visualizations saved to {vis_dir}")
    print(f"Metrics saved to {metrics_path}")
    
    return metrics, avg_metrics

def main():
    parser = argparse.ArgumentParser(description="Run MedSAM2 inference on Mpox images")
    parser.add_argument("--input_dir", required=True, help="Directory containing images or npz files")
    parser.add_argument("--output_dir", default="./mpox_results", help="Output directory for results")
    parser.add_argument("--sam2_checkpoint", required=True, help="Path to SAM2 checkpoint")
    parser.add_argument("--medsam2_checkpoint", default=None, help="Path to MedSAM2 fine-tuned checkpoint")
    parser.add_argument("--model_cfg", default="sam2.1_hiera_b+.yaml", help="Path or name of model config file")
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
    metrics, avg_metrics = run_inference(
        args.input_dir,
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
