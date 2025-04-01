#!/usr/bin/env python3
"""
Simple script to convert COCO format annotations to binary masks.
This version avoids relying on pycocotools for RLE decoding.
"""

import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

def ensure_dir(directory):
    """Make sure directory exists."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory

def coco_to_binary_masks(coco_json_path, img_dir, output_dir, visualize=False):
    """Convert COCO format annotations to binary masks."""
    # Load COCO annotations
    print(f"Loading COCO annotations from {coco_json_path}")
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    print(f"COCO data keys: {list(coco_data.keys())}")
    print(f"Found {len(coco_data.get('images', []))} images")
    print(f"Found {len(coco_data.get('annotations', []))} annotations")
    print(f"Found {len(coco_data.get('categories', []))} categories")
    
    # Create mapping from image ID to filename
    image_id_to_filename = {}
    image_id_to_info = {}
    for image in coco_data.get('images', []):
        image_id = image.get('id')
        if image_id is not None:
            image_id_to_filename[image_id] = image.get('file_name')
            image_id_to_info[image_id] = image
    
    # Create mapping from image ID to annotations
    image_id_to_annotations = {}
    for ann in coco_data.get('annotations', []):
        image_id = ann.get('image_id')
        if image_id is not None:
            if image_id not in image_id_to_annotations:
                image_id_to_annotations[image_id] = []
            image_id_to_annotations[image_id].append(ann)
    
    # Create output directory for masks
    ensure_dir(output_dir)
    if visualize:
        vis_dir = ensure_dir(os.path.join(output_dir, "visualization"))
    
    # Print a sample annotation to understand the structure
    if coco_data.get('annotations'):
        sample_ann = coco_data['annotations'][0]
        print(f"Sample annotation: {json.dumps(sample_ann, indent=2)}")
    
    # Process each image
    print(f"Processing {len(image_id_to_filename)} images...")
    for image_id, filename in tqdm(image_id_to_filename.items()):
        # Try to find the image file
        img_path = os.path.join(img_dir, filename)
        img = None
        
        try:
            img = cv2.imread(img_path)
            if img is None:
                # Try looking for the image without path
                alt_path = os.path.join(img_dir, os.path.basename(filename))
                img = cv2.imread(alt_path)
                if img is None:
                    print(f"Warning: Could not read image {img_path} or {alt_path}")
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
        
        # Get image dimensions
        if img is not None:
            height, width = img.shape[:2]
        else:
            # Try to get dimensions from COCO data
            image_info = image_id_to_info.get(image_id, {})
            height = image_info.get('height', 512)  # Default if not found
            width = image_info.get('width', 512)    # Default if not found
            print(f"Using dimensions from COCO: {width}x{height}")
        
        # Create an empty mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Get annotations for this image
        annotations = image_id_to_annotations.get(image_id, [])
        if not annotations:
            print(f"Warning: No annotations found for image {filename}")
        
        # Process each annotation
        for ann in annotations:
            try:
                # If polygon segmentation is available, use it
                if 'segmentation' in ann and isinstance(ann['segmentation'], list) and ann['segmentation']:
                    for segment in ann['segmentation']:
                        if isinstance(segment, list) and len(segment) >= 6:  # At least 3 points
                            try:
                                # Convert polygon to mask
                                poly = np.array(segment).reshape(-1, 2)
                                cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
                            except Exception as e:
                                print(f"Error processing polygon for {filename}: {e}")
                
                # If bounding box is available, use it as fallback
                elif 'bbox' in ann and isinstance(ann['bbox'], list) and len(ann['bbox']) >= 4:
                    try:
                        x, y, w, h = map(int, ann['bbox'])
                        cv2.rectangle(mask, (x, y), (x+w, y+h), 1, thickness=-1)  # Fill rectangle
                    except Exception as e:
                        print(f"Error processing bbox for {filename}: {e}")
                
                else:
                    print(f"Warning: No usable annotation format found for annotation {ann.get('id')} in {filename}")
            
            except Exception as e:
                print(f"Error processing annotation for {filename}: {e}")
        
        # If mask is empty, create a default central region
        if np.sum(mask) == 0:
            print(f"Warning: Empty mask for {filename}, creating default central region")
            center_x, center_y = width // 2, height // 2
            radius = min(width, height) // 4
            cv2.circle(mask, (center_x, center_y), radius, 1, thickness=-1)
        
        # Save the mask
        mask_filename = os.path.splitext(os.path.basename(filename))[0] + "_mask.png"
        mask_path = os.path.join(output_dir, mask_filename)
        cv2.imwrite(mask_path, mask * 255)
        print(f"Saved mask to {mask_path}")
        
        # Visualize if requested
        if visualize and img is not None:
            plt.figure(figsize=(15, 5))
            
            # Original image
            plt.subplot(1, 3, 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title("Original Image")
            plt.axis("off")
            
            # Mask
            plt.subplot(1, 3, 2)
            plt.imshow(mask, cmap='gray')
            plt.title("Mask")
            plt.axis("off")
            
            # Overlay
            plt.subplot(1, 3, 3)
            overlay = img.copy()
            overlay[mask == 1] = (0, 0, 255)  # Red overlay
            plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            plt.title("Overlay")
            plt.axis("off")
            
            plt.tight_layout()
            vis_path = os.path.join(vis_dir, os.path.splitext(os.path.basename(filename))[0] + "_vis.png")
            plt.savefig(vis_path)
            plt.close()
    
    print(f"Conversion completed. Masks saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Convert COCO format annotations to binary masks")
    parser.add_argument("--coco_json", required=True, help="Path to COCO format JSON file")
    parser.add_argument("--img_dir", required=True, help="Directory containing the images")
    parser.add_argument("--output_dir", default="./masks", help="Output directory for masks")
    parser.add_argument("--visualize", action="store_true", help="Create visualization of masks")
    
    args = parser.parse_args()
    
    coco_to_binary_masks(args.coco_json, args.img_dir, args.output_dir, args.visualize)

if __name__ == "__main__":
    main()
