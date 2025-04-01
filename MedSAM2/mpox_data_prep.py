#!/usr/bin/env python3
"""
Data preparation script for Mpox lesion segmentation with MedSAM2.
This script prepares Mpox image data into the format required by MedSAM2.
"""

import os
import sys
import numpy as np
import argparse
from pathlib import Path
import cv2
import shutil
from tqdm import tqdm
import json
import random
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import matplotlib.pyplot as plt

def ensure_dir(directory):
    """Make sure directory exists."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory

def find_images(directory, extensions=['.jpg', '.jpeg', '.png', '.bmp']):
    """Find all images in a directory with specified extensions."""
    image_files = []
    for ext in extensions:
        image_files.extend(list(Path(directory).glob(f"**/*{ext}")))
    return sorted(image_files)

def preprocess_image(image, target_size=(1024, 1024)):
    """Preprocess image for MedSAM2 model."""
    # Convert BGR to RGB if needed
    if image.shape[2] == 3:
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
    
    return canvas

def process_image_for_inference(image_path, output_dir, target_size=(1024, 1024)):
    """Process a single image for inference with MedSAM2."""
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error reading image: {image_path}")
        return None
    
    # Preprocess the image
    processed_image = preprocess_image(image, target_size)
    
    # Get original dimensions
    orig_h, orig_w = image.shape[:2]
    
    # Create filename for npz
    stem = image_path.stem
    npz_filename = os.path.join(output_dir, f"{stem}.npz")
    
    # Save image and metadata to npz
    np.savez_compressed(
        npz_filename,
        image=processed_image,
        orig_size=np.array([orig_h, orig_w]),
        filename=str(image_path.name)
    )
    
    return npz_filename

def process_image_and_mask(image_path, mask_path, output_dir, target_size=(1024, 1024), down_size=(256, 256), split='train'):
    """Process an image and its mask for training with MedSAM2."""
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error reading image: {image_path}")
        return None
    
    # Preprocess the image
    processed_image = preprocess_image(image, target_size)
    
    # Determine mask path if not provided directly
    if mask_path is None:
        # Try to find a matching mask
        potential_mask_paths = [
            image_path.with_name(f"{image_path.stem}_mask{image_path.suffix}"),
            image_path.with_name(f"{image_path.stem}_mask.png"),
            image_path.with_name(f"{image_path.stem}_segmentation{image_path.suffix}"),
            image_path.with_name(f"{image_path.stem}_seg.png"),
        ]
        
        for potential_path in potential_mask_paths:
            if potential_path.exists():
                mask_path = potential_path
                break
        
        if mask_path is None:
            print(f"No matching mask found for {image_path}")
            return None
    
    # Load mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error reading mask: {mask_path}")
        return None
    
    # Binarize mask if needed
    mask = (mask > 0).astype(np.uint8)
    
    # Resize mask to match the processed image size
    h, w = image.shape[:2]
    ratio = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * ratio), int(w * ratio)
    
    mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # Create a black canvas of target size
    mask_canvas = np.zeros(target_size, dtype=np.uint8)
    
    # Paste the resized mask onto the canvas (center it)
    y_offset = (target_size[0] - new_h) // 2
    x_offset = (target_size[1] - new_w) // 2
    mask_canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = mask_resized
    
    # Get original dimensions
    orig_h, orig_w = image.shape[:2]
    
    # For training data, create both npz and npy formats
    npz_dir = os.path.join(output_dir, f"npz_{split}")
    ensure_dir(npz_dir)
    
    # Create filename for npz
    stem = image_path.stem
    npz_filename = os.path.join(npz_dir, f"{stem}.npz")
    
    # Calculate bounding box from mask
    if mask_canvas.sum() > 0:  # If the mask is not empty
        y_indices, x_indices = np.where(mask_canvas > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        bbox = np.array([x_min, y_min, x_max, y_max])
    else:
        # Handle empty masks by using a default small box in the center
        h, w = target_size
        center_x, center_y = w // 2, h // 2
        size = min(h, w) // 10
        bbox = np.array([center_x - size, center_y - size, center_x + size, center_y + size])
    
    # Save image, mask, and metadata to npz
    np.savez_compressed(
        npz_filename,
        image=processed_image,
        mask=mask_canvas,
        bbox=bbox,
        orig_size=np.array([orig_h, orig_w]),
        filename=str(image_path.name)
    )
    
    # For training split, also save as npy files (downsized version for training efficiency)
    if split == 'train':
        npy_dir = os.path.join(output_dir, "npy")
        ensure_dir(os.path.join(npy_dir, "imgs"))
        ensure_dir(os.path.join(npy_dir, "gts"))
        
        # Create downsized (256x256) versions for training efficiency
        img_down = cv2.resize(processed_image, down_size, interpolation=cv2.INTER_AREA)
        mask_down = cv2.resize(mask_canvas, down_size, interpolation=cv2.INTER_NEAREST)
        
        # Save image and mask as separate npy files
        np.save(os.path.join(npy_dir, "imgs", f"{stem}.npy"), img_down)
        np.save(os.path.join(npy_dir, "gts", f"{stem}.npy"), mask_down)
    
    return npz_filename

def prepare_data_for_inference(image_dir, output_dir, num_workers=4, target_size=(1024, 1024)):
    """Prepare the data for inference with MedSAM2."""
    # Find all images
    image_files = find_images(image_dir)
    print(f"Found {len(image_files)} images for inference.")
    
    if len(image_files) == 0:
        print(f"No images found in {image_dir}")
        return
    
    # Create output directory for npz files
    npz_dir = os.path.join(output_dir, "npz_inference")
    ensure_dir(npz_dir)
    
    # Process images in parallel
    process_func = partial(process_image_for_inference, output_dir=npz_dir, target_size=target_size)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(process_func, image_files),
            total=len(image_files),
            desc="Processing images for inference"
        ))
    
    # Count successful conversions
    successful = [r for r in results if r is not None]
    print(f"Successfully processed {len(successful)}/{len(image_files)} images.")
    print(f"Prepared inference data saved to {npz_dir}")
    
    return npz_dir

def prepare_data_for_training(image_dir, mask_dir, output_dir, 
                            val_ratio=0.2, num_workers=4, target_size=(1024, 1024), down_size=(256, 256)):
    """Prepare the data for training/fine-tuning MedSAM2."""
    # Find all images
    image_files = find_images(image_dir)
    print(f"Found {len(image_files)} images for training.")
    
    if len(image_files) == 0:
        print(f"No images found in {image_dir}")
        return
    
    # Create output directories
    ensure_dir(output_dir)
    
    # Split into train and validation
    random.seed(42)  # For reproducibility
    random.shuffle(image_files)
    
    val_count = max(1, int(len(image_files) * val_ratio))
    val_files = image_files[:val_count]
    train_files = image_files[val_count:]
    
    print(f"Split dataset: {len(train_files)} for training, {len(val_files)} for validation")
    
    # Find masks for all images
    if mask_dir:
        # If a specific mask directory is provided
        mask_files = find_images(mask_dir, extensions=['.png', '.jpg', '.jpeg', '.bmp'])
        
        # Create a mapping from image stem to mask path
        mask_map = {}
        for m in mask_files:
            # Try different stem variations to match with original images
            mask_stem = m.stem
            if "_mask" in mask_stem:
                orig_stem = mask_stem.replace("_mask", "")
                mask_map[orig_stem] = m
            else:
                mask_map[mask_stem] = m
        
        # Assign mask to each image
        image_mask_pairs_train = []
        for img in train_files:
            mask = mask_map.get(img.stem)
            if mask:
                image_mask_pairs_train.append((img, mask))
            else:
                print(f"No mask found for {img.stem}")
        
        image_mask_pairs_val = []
        for img in val_files:
            mask = mask_map.get(img.stem)
            if mask:
                image_mask_pairs_val.append((img, mask))
            else:
                print(f"No mask found for {img.stem}")
    else:
        # If masks are expected to be alongside images or follow a naming pattern
        image_mask_pairs_train = [(img, None) for img in train_files]
        image_mask_pairs_val = [(img, None) for img in val_files]
    
    # Process training images and masks
    train_process_func = partial(
        process_image_and_mask, 
        output_dir=output_dir, 
        target_size=target_size,
        down_size=down_size,
        split='train'
    )
    
    print("Processing training images and masks...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        train_results = list(tqdm(
            executor.map(train_process_func, 
                        [pair[0] for pair in image_mask_pairs_train],
                        [pair[1] for pair in image_mask_pairs_train]),
            total=len(image_mask_pairs_train),
            desc="Processing training data"
        ))
    
    # Process validation images and masks
    val_process_func = partial(
        process_image_and_mask, 
        output_dir=output_dir, 
        target_size=target_size,
        down_size=down_size,
        split='val'
    )
    
    print("Processing validation images and masks...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        val_results = list(tqdm(
            executor.map(val_process_func, 
                        [pair[0] for pair in image_mask_pairs_val],
                        [pair[1] for pair in image_mask_pairs_val]),
            total=len(image_mask_pairs_val),
            desc="Processing validation data"
        ))
    
    # Count successful conversions
    successful_train = [r for r in train_results if r is not None]
    successful_val = [r for r in val_results if r is not None]
    
    print(f"Successfully processed {len(successful_train)}/{len(image_mask_pairs_train)} training images.")
    print(f"Successfully processed {len(successful_val)}/{len(image_mask_pairs_val)} validation images.")
    print(f"Prepared training data saved to {output_dir}")
    
    npz_train_dir = os.path.join(output_dir, "npz_train")
    npz_val_dir = os.path.join(output_dir, "npz_val")
    npy_dir = os.path.join(output_dir, "npy")
    
    return npz_train_dir, npz_val_dir, npy_dir

def visualize_samples(npz_dir, output_dir, num_samples=5):
    """Visualize some sample images and masks for verification."""
    # Find all npz files
    npz_files = list(Path(npz_dir).glob("*.npz"))
    
    if len(npz_files) == 0:
        print(f"No npz files found in {npz_dir}")
        return
    
    # Select random samples
    samples = random.sample(npz_files, min(num_samples, len(npz_files)))
    
    # Create output directory
    vis_dir = os.path.join(output_dir, "visualization")
    ensure_dir(vis_dir)
    
    for i, sample_path in enumerate(samples):
        # Load npz file
        data = np.load(sample_path, allow_pickle=True)
        
        # Create figure
        plt.figure(figsize=(15, 5))
        
        # Plot image
        plt.subplot(1, 3, 1)
        plt.imshow(data['image'])
        plt.title("Original Image")
        plt.axis('off')
        
        # Plot mask if available
        if 'mask' in data:
            plt.subplot(1, 3, 2)
            plt.imshow(data['mask'], cmap='gray')
            plt.title("Mask")
            plt.axis('off')
            
            # Plot overlay
            plt.subplot(1, 3, 3)
            overlay = data['image'].copy()
            mask_rgb = np.zeros_like(overlay)
            mask_rgb[:,:,0] = data['mask'] * 255  # Red channel
            overlay = cv2.addWeighted(overlay, 0.7, mask_rgb, 0.3, 0)
            plt.imshow(overlay)
            plt.title("Overlay")
            plt.axis('off')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"sample_{i+1}.png"))
        plt.close()
    
    print(f"Saved {len(samples)} sample visualizations to {vis_dir}")

def main():
    parser = argparse.ArgumentParser(description="Prepare Mpox data for MedSAM2")
    parser.add_argument("--image_dir", required=True, help="Directory containing Mpox images")
    parser.add_argument("--mask_dir", default=None, help="Directory containing mask images (if separate)")
    parser.add_argument("--output_dir", default="./mpox_data", help="Output directory for processed data")
    parser.add_argument("--mode", choices=["inference", "training"], default="inference",
                        help="Prepare data for inference or training/fine-tuning")
    parser.add_argument("--val_ratio", type=float, default=0.2, 
                        help="Validation set ratio (for training mode)")
    parser.add_argument("--target_size", type=int, nargs=2, default=[1024, 1024],
                        help="Target size for processing images (height width)")
    parser.add_argument("--down_size", type=int, nargs=2, default=[256, 256],
                        help="Size for training npy files (height width)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of parallel workers")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations of processed samples")
    
    args = parser.parse_args()
    
    # Set up output directory
    ensure_dir(args.output_dir)
    
    # Process based on mode
    if args.mode == "inference":
        npz_dir = prepare_data_for_inference(
            args.image_dir, 
            args.output_dir,
            args.num_workers,
            tuple(args.target_size)
        )
        
        if args.visualize and npz_dir:
            visualize_samples(npz_dir, args.output_dir)
    else:
        npz_train_dir, npz_val_dir, npy_dir = prepare_data_for_training(
            args.image_dir,
            args.mask_dir,
            args.output_dir,
            args.val_ratio,
            args.num_workers,
            tuple(args.target_size),
            tuple(args.down_size)
        )
        
        if args.visualize:
            # Visualize some training samples
            visualize_samples(npz_train_dir, args.output_dir)
    
    # Save configuration for reference
    config = vars(args)
    from datetime import datetime
    config['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(os.path.join(args.output_dir, "prep_config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Data preparation completed successfully!")

if __name__ == "__main__":
    main()
