#!/usr/bin/env python3
"""
Parameter optimization for Mpox lesion detection.
This script helps find the optimal parameters for accurate lesion detection.
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
import json

# Import our detector class
from headless_lesion_detector import HeadlessLesionDetector

def evaluate_params(image_path, min_area, max_area, circularity_thresh, convexity_thresh):
    """
    Test detection parameters on a single image and display results.

    Args:
        image_path: Path to test image
        min_area: Minimum lesion area
        max_area: Maximum lesion area
        circularity_thresh: Circularity threshold (0-1)
        convexity_thresh: Convexity threshold (0-1)

    Returns:
        Dictionary with detection results
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None

    # Create a customized detect_lesions function with the specified parameters
    def custom_detect_lesions(img):
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Use multiple thresholding methods and combine results
        _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        adapt_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 11, 2)

        # Color-based thresholding
        s_channel = hsv[:, :, 1]
        _, sat_thresh = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Combine thresholding results
        combined_thresh = cv2.bitwise_or(otsu_thresh, adapt_thresh)

        # Clean up with morphological operations
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        opening = cv2.morphologyEx(combined_thresh, cv2.MORPH_OPEN, kernel_small)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_medium)

        # Find contours
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area, circularity, and convexity
        valid_contours = []
        areas = []

        for contour in contours:
            # Calculate contour area
            area = cv2.contourArea(contour)

            # Skip if area is too small or too large
            if area < min_area or area > max_area:
                continue

            # Calculate perimeter
            perimeter = cv2.arcLength(contour, True)

            # Calculate circularity: 4*pi*area/perimeter^2 (1.0 = perfect circle)
            circularity = (4 * np.pi * area) / (perimeter * perimeter + 1e-6)

            # Calculate convexity: area / convex hull area
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            convexity = area / (hull_area + 1e-6)

            # Filter based on shape metrics
            if circularity > circularity_thresh and convexity > convexity_thresh:
                valid_contours.append(contour)
                areas.append(area)

        # Create masks for valid contours
        mask = np.zeros_like(gray)

        for contour in valid_contours:
            # Draw filled contour
            cv2.drawContours(mask, [contour], -1, 255, -1)

        # Create bounding boxes
        boxes = []
        for contour in valid_contours:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append([x, y, x + w, y + h])

        # Return detection results
        return {
            'lesion_count': len(valid_contours),
            'contours': valid_contours,
            'areas': areas,
            'total_area': sum(areas),
            'mask': mask,
            'boxes': boxes
        }

    # Run detection with custom parameters
    detections = custom_detect_lesions(image)

    # Create visualization
    vis_image = image.copy()

    # Draw each lesion with a unique color
    for i, (contour, area) in enumerate(zip(detections['contours'], detections['areas'])):
        # Generate a consistent color for this lesion
        color_value = (i * 35) % 180 + 40
        color = (0, color_value, 255 - color_value)  # BGR format

        # Draw the contour
        cv2.drawContours(vis_image, [contour], -1, color, 2)

        # Draw lesion ID and metrics
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Calculate perimeter and shape metrics for display
            perimeter = cv2.arcLength(contour, True)
            circularity = (4 * np.pi * area) / (perimeter * perimeter + 1e-6)

            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            convexity = area / (hull_area + 1e-6)

            # Draw lesion ID
            cv2.putText(vis_image, f"#{i+1}", (cX-15, cY),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw metrics
            cv2.putText(vis_image, f"Circ: {circularity:.2f}", (cX-40, cY+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(vis_image, f"Conv: {convexity:.2f}", (cX-40, cY+40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw summary
    summary = f"Lesions: {detections['lesion_count']}, Params: minArea={min_area}, circ={circularity_thresh}, conv={convexity_thresh}"
    cv2.putText(vis_image, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return {
        'detections': detections,
        'visualization': vis_image
    }

def optimize_params(image_paths, output_dir):
    """
    Find optimal parameters by testing various combinations

    Args:
        image_paths: List of image paths to test
        output_dir: Directory to save results
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Parameter ranges to test
    min_areas = [50, 100, 150, 200]
    circularity_thresholds = [0.2, 0.3, 0.4, 0.5]
    convexity_thresholds = [0.6, 0.7, 0.8, 0.9]

    # Fixed max area
    max_area = 50000

    # Track best parameters
    best_params = None
    best_avg_count = 0

    # Results for each parameter combination
    results = []

    # Test each parameter combination on all images
    total_combinations = len(min_areas) * len(circularity_thresholds) * len(convexity_thresholds)
    combination_count = 0

    for min_area in min_areas:
        for circ_thresh in circularity_thresholds:
            for conv_thresh in convexity_thresholds:
                combination_count += 1
                print(f"Testing combination {combination_count}/{total_combinations}: " +
                      f"min_area={min_area}, circ={circ_thresh}, conv={conv_thresh}")

                # Process each image with current parameters
                total_count = 0
                for i, img_path in enumerate(image_paths):
                    result = evaluate_params(img_path, min_area, max_area, circ_thresh, conv_thresh)

                    if result:
                        count = result['detections']['lesion_count']
                        total_count += count

                        # Save visualization for the first few images
                        if i < 3:
                            out_path = os.path.join(output_dir,
                                                   f"param_test_{min_area}_{circ_thresh}_{conv_thresh}_{i}.png")
                            cv2.imwrite(out_path, result['visualization'])

                # Calculate average lesion count
                avg_count = total_count / len(image_paths) if image_paths else 0

                # Record results
                param_result = {
                    'min_area': min_area,
                    'max_area': max_area,
                    'circularity_threshold': circ_thresh,
                    'convexity_threshold': conv_thresh,
                    'average_lesion_count': avg_count
                }
                results.append(param_result)

                # Check if this is the best so far
                if avg_count > best_avg_count:
                    best_avg_count = avg_count
                    best_params = param_result

    # Save all results
    with open(os.path.join(output_dir, 'parameter_optimization_results.json'), 'w') as f:
        json.dump({
            'all_results': results,
            'best_params': best_params
        }, f, indent=2)

    # Report best parameters
    print("\nOptimization Complete!")
    print(f"Best parameters: min_area={best_params['min_area']}, " +
          f"circularity={best_params['circularity_threshold']}, " +
          f"convexity={best_params['convexity_threshold']}")
    print(f"Average lesion count: {best_params['average_lesion_count']:.2f}")

    return best_params

def main():
    parser = argparse.ArgumentParser(description="Optimize parameters for Mpox lesion detection")

    # Input and output
    parser.add_argument('--input', required=True, help='Path to input image or directory')
    parser.add_argument('--output_dir', default='./param_optimization', help='Directory to save outputs')

    # Specific parameter testing (if not optimizing)
    parser.add_argument('--min_area', type=int, default=100, help='Minimum lesion area in pixels')
    parser.add_argument('--max_area', type=int, default=50000, help='Maximum lesion area in pixels')
    parser.add_argument('--circularity', type=float, default=0.3, help='Circularity threshold (0-1)')
    parser.add_argument('--convexity', type=float, default=0.7, help='Convexity threshold (0-1)')

    # Optimization options
    parser.add_argument('--optimize', action='store_true', help='Run parameter optimization')
    parser.add_argument('--test_images', type=int, default=10, help='Number of images to test during optimization')

    args = parser.parse_args()

    # Create output directory if needed
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Collect image paths
    image_paths = []

    if os.path.isdir(args.input):
        # Find all images in the directory
        for ext in ('.jpg', '.jpeg', '.png'):
            image_paths.extend(list(Path(args.input).glob(f"**/*{ext}")))

        print(f"Found {len(image_paths)} images in {args.input}")
    else:
        # Single image
        image_paths = [args.input]

    if args.optimize:
        # Limit number of test images for optimization
        if len(image_paths) > args.test_images:
            print(f"Using {args.test_images} images for optimization")
            # Use a representative sample (e.g., every nth image)
            step = len(image_paths) // args.test_images
            image_paths = image_paths[::step][:args.test_images]

        # Run optimization
        best_params = optimize_params(image_paths, args.output_dir)
    else:
        # Test specific parameters on all images
        for i, img_path in enumerate(image_paths):
            result = evaluate_params(
                img_path, args.min_area, args.max_area,
                args.circularity, args.convexity
            )

            if result:
                out_path = os.path.join(args.output_dir, f"detection_{i}.png")
                cv2.imwrite(out_path, result['visualization'])

                print(f"Processed {img_path}")
                print(f"Detected {result['detections']['lesion_count']} lesions")
                print(f"Total area: {result['detections']['total_area']:.0f} pixels")
                print(f"Visualization saved to {out_path}")
                print()

if __name__ == "__main__":
    main()
