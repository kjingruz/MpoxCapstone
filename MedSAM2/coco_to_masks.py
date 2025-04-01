#!/usr/bin/env python3
"""
Convert COCO format annotations to binary masks for MedSAM2 fine-tuning.
This script extracts segmentation masks from COCO format and saves them as PNG files.
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
from pycocotools import mask as coco_mask

def ensure_dir(directory):
    """Make sure directory exists."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory

def coco_to_binary_masks(coco_json_path, img_dir, output_dir, visualize=False):
    """Convert COCO format annotations to binary masks."""
    # Load COCO annotations
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create mapping from image ID to filename
    image_id_to_filename = {}
    image_id_to_size = {}  # To store image dimensions
    for image in coco_data['images']:
        image_id_to_filename[image['id']] = image['file_name']
        image_id_to_size[image['id']] = (image['height'], image['width'])
    
    # Create mapping from image ID to annotations
    image_id_to_annotations = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_id_to_annotations:
            image_id_to_annotations[image_id] = []
        image_id_to_annotations[image_id].append(ann)
    
    # Create output directory for masks
    ensure_dir(output_dir)
    if visualize:
        vis_dir = ensure_dir(os.path.join(output_dir, "visualization"))
    
    # Process each image
    print(f"Processing {len(coco_data['images'])} images...")
    for image in tqdm(coco_data['images']):
        image_id = image['id']
        filename = image['file_name']
        
        # Get image dimensions from COCO data
        height, width = image_id_to_size[image_id]
        
        # Create an empty mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Get annotations for this image
        annotations = image_id_to_annotations.get(image_id, [])
        
        # Draw each segmentation on the mask
        for ann in annotations:
            if 'segmentation' in ann:
                # Handle RLE format
                if isinstance(ann['segmentation'], dict) or (isinstance(ann['segmentation'], list) and len(ann['segmentation']) > 0 and isinstance(ann['segmentation'][0], dict)):
                    try:
                        # This is likely an RLE encoded mask
                        if isinstance(ann['segmentation'], dict):
                            rle = ann['segmentation']
                        else:
                            rle = ann['segmentation'][0]  # Take the first one if it's a list
                        
                        # Directly convert using pycocotools if format is correct
                        if 'counts' in rle and 'size' in rle:
                            rle_obj = {
                                'counts': rle['counts'],
                                'size': rle['size']
                            }
                            binary_mask = coco_mask.decode(rle_obj)
                            mask = np.logical_or(mask, binary_mask).astype(np.uint8)
                    except Exception as e:
                        print(f"Error decoding RLE for annotation {ann['id']}: {e}")
                        # Fall back to using the bounding box if available
                        if 'bbox' in ann:
                            bbox = ann['bbox']  # [x, y, width, height]
                            x, y, w, h = map(int, bbox)
                            # Fill a rectangular mask for this bbox
                            mask[y:y+h, x:x+w] = 1
                
                # Handle polygon format
                elif isinstance(ann['segmentation'], list):
                    for seg in ann['segmentation']:
                        if isinstance(seg, list) and len(seg) >= 6:  # Min 3 points (x,y) for a polygon
                            # Convert polygon to mask
                            try:
                                poly = np.array(seg).reshape(-1, 2)
                                cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
                            except Exception as e:
                                print(f"Error with polygon annotation {ann['id']}: {e}")
            
            # Fallback to bbox if segmentation fails or is not available
            elif 'bbox' in ann:
                bbox = ann['bbox']  # [x, y, width, height]
                x, y, w, h = map(int, bbox)
                # Fill a rectangular mask for this bbox
                mask[y:y+h, x:x+w] = 1
        
        # Read the actual image to verify and for visualization
        img_path = os.path.join(img_dir, filename)
        img = None
        if visualize or not mask.any():  # Only read image if we need visualization or mask is empty
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Could not read image {img_path}")
            else:
                print(f"Warning: Image {img_path} not found")
        
        # Fallback for empty masks - create a central region
        if not mask.any() and img is not None:
            h, w = img.shape[:2]
            center_y, center_x = h // 2, w // 2
            radius = min(h, w) // 4
            Y, X = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            mask = (dist_from_center <= radius).astype(np.uint8)
            print(f"Warning: Created fallback mask for {filename} as no valid segmentation found")
        
        # Save the mask
        mask_filename = os.path.splitext(filename)[0] + "_mask.png"
        cv2.imwrite(os.path.join(output_dir, mask_filename), mask * 255)
        
        # Visualize mask overlay if requested
        if visualize and img is not None:
            plt.figure(figsize=(12, 6))
            
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
            overlay[mask == 1] = (0, 0, 255)  # Red overlay for segmentation
            plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            plt.title("Overlay")
            plt.axis("off")
            
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, os.path.splitext(filename)[0] + "_vis.png"))
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
