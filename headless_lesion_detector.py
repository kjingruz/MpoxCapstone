#!/usr/bin/env python3
"""
Headless Mpox Lesion Detector using OpenCV Headless.
No GUI dependencies required.
"""

import os
import sys
import argparse
import numpy as np
import cv2
import json
from datetime import datetime
from pathlib import Path

class HeadlessLesionDetector:
    """
    Headless lesion detector using classical computer vision techniques.
    Works with OpenCV headless without GUI dependencies.
    """
    def __init__(self, min_area=50, max_area=100000, threshold_method='otsu'):
        """
        Initialize the detector with parameters.
    
        Args:
            min_area: Minimum lesion area in pixels
            max_area: Maximum lesion area in pixels
            threshold_method: Method for thresholding ('otsu', 'adaptive', 'simple')
        """
        self.min_area = min_area
        self.max_area = max_area
        self.threshold_method = threshold_method

    def detect_mpox_lesions(self, image):
        """
        Specialized lesion detection algorithm targeting Mpox-specific characteristics.
        Focuses on darker, circular lesions against skin background.
        
        Args:
            image: Input image (numpy array, BGR format)
            
        Returns:
            Dictionary with detection results
        """
        # Create a copy of the original image
        orig_image = image.copy()
        
        # STEP 1: Prepare different color representations
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # STEP 2: Extract skin tone and create a relative darkness map
        # Extract value channel (brightness) from HSV
        _, _, v_channel = cv2.split(hsv)
        
        # Blur to get average skin tone (background)
        blurred_v = cv2.GaussianBlur(v_channel, (51, 51), 0)
        
        # Create darkness map (how much darker each pixel is compared to surrounding skin)
        # Higher values indicate darker spots compared to surrounding skin
        darkness_map = cv2.subtract(blurred_v, v_channel)
        
        # STEP 3: Local contrast enhancement to make lesions stand out
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        darkness_enhanced = clahe.apply(darkness_map)
        
        # STEP 4: Adaptive thresholding to identify darker regions
        # Use regional information rather than global thresholds
        dark_binary = cv2.adaptiveThreshold(
            darkness_enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            -2  # Negative C value to detect darker regions
        )
        
        # STEP 5: Create a second mask based on saturation (often higher in lesions)
        s_channel = hsv[:, :, 1]
        s_enhanced = clahe.apply(s_channel)
        _, sat_binary = cv2.threshold(s_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # STEP 6: Clean up masks with morphological operations
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Remove small noise
        dark_clean = cv2.morphologyEx(dark_binary, cv2.MORPH_OPEN, kernel_small)
        
        # Fill small holes
        dark_filled = cv2.morphologyEx(dark_clean, cv2.MORPH_CLOSE, kernel_medium)
        
        # STEP 7: Find contours in the mask
        contours, _ = cv2.findContours(dark_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # STEP 8: Filter contours with stricter criteria specific to Mpox lesions
        valid_contours = []
        areas = []
        
        for contour in contours:
            # Calculate area
            area = cv2.contourArea(contour)
            
            # Filter by area - be more specific about lesion size
            # Mpox lesions are typically not extremely large or extremely small
            if area < self.min_area or area > 10000:  # Stricter upper limit
                continue
            
            # Calculate perimeter and derived shape metrics
            perimeter = cv2.arcLength(contour, True)
            
            # Calculate circularity: 4*pi*area/perimeter^2 (1.0 = perfect circle)
            # Mpox lesions are often more circular than other skin features
            circularity = (4 * np.pi * area) / (perimeter * perimeter + 1e-6)
            
            # More stringent circularity check
            if circularity < 0.3:  # Require more circular shapes
                continue
                
            # Calculate aspect ratio using bounding rect
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / (h + 1e-6)
            
            # Mpox lesions are rarely very elongated
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue
            
            # Check average darkness inside the contour
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_darkness = cv2.mean(darkness_map, mask=mask)[0]
            
            # Only keep regions that are genuinely darker than surroundings
            if mean_darkness < 10:  # Threshold for darkness
                continue
                
            # If it passed all filters, keep it
            valid_contours.append(contour)
            areas.append(area)
        
        # STEP 9: Create mask and bounding boxes
        mask = np.zeros_like(gray)
        
        for contour in valid_contours:
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

    def detect_lesions(self, image):
        """
        Enhanced lesion detection algorithm for Mpox lesions.
        Uses multiple color spaces and techniques to better detect lesions.
        
        Args:
            image: Input image (numpy array, BGR format)
            
        Returns:
            Dictionary with detection results
        """
        # Create a copy of the original image
        orig_image = image.copy()
        
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # ENHANCEMENT 1: Contrast enhancement
        # Apply CLAHE for better local contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(gray)
        
        # ENHANCEMENT 2: Multi-scale Gaussian filtering
        # Blur at different scales to capture lesions of different sizes
        blur1 = cv2.GaussianBlur(gray_clahe, (3, 3), 0)
        blur2 = cv2.GaussianBlur(gray_clahe, (7, 7), 0)
        blur_diff = cv2.absdiff(blur1, blur2)  # Difference of Gaussians
        
        # ENHANCEMENT 3: Extract color channel information specific to lesions
        # Extract a and b channels from LAB (more robust color information)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Extract V (value) and S (saturation) from HSV
        h_channel, s_channel, v_channel = cv2.split(hsv)
        
        # ENHANCEMENT 4: Create a "redness map" (enhances reddish lesions)
        b, g, r = cv2.split(image)
        redness = cv2.subtract(r, (g + b) // 2)  # Higher values for reddish areas
        
        # ENHANCEMENT 5: Multi-threshold approach
        # Apply different thresholding methods
        
        # 5.1 Otsu on grayscale
        _, otsu_thresh = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 5.2 Adaptive thresholding
        adapt_thresh = cv2.adaptiveThreshold(blur1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 11, 2)
        
        # 5.3 Threshold on redness
        _, redness_thresh = cv2.threshold(redness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 5.4 Threshold on a_channel (redness in LAB space)
        _, a_thresh = cv2.threshold(a_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 5.5 Threshold on saturation
        _, s_thresh = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 5.6 Combine thresholds
        combined_thresh1 = cv2.bitwise_or(otsu_thresh, adapt_thresh)
        combined_thresh2 = cv2.bitwise_or(redness_thresh, a_thresh)
        combined_thresh3 = cv2.bitwise_or(combined_thresh1, combined_thresh2)
        combined_thresh = cv2.bitwise_or(combined_thresh3, s_thresh)
        
        # ENHANCEMENT 6: Improved morphological operations
        # Create kernels for morphological operations
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        
        # 6.1 Opening to remove small noise
        opening = cv2.morphologyEx(combined_thresh, cv2.MORPH_OPEN, kernel_small)
        
        # 6.2 Closing to fill holes inside lesions
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_medium)
        
        # 6.3 Use tophat to enhance small, bright features (for smaller lesions)
        tophat = cv2.morphologyEx(gray_clahe, cv2.MORPH_TOPHAT, kernel_large)
        _, tophat_thresh = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 6.4 Combine with tophat to catch smaller lesions
        final_mask = cv2.bitwise_or(closing, tophat_thresh)
        
        # ENHANCEMENT 7: Watershed to separate touching lesions
        # Distance transform
        dist_transform = cv2.distanceTransform(final_mask, cv2.DIST_L2, 3)
        
        # Threshold to get markers
        _, markers = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, cv2.THRESH_BINARY)
        markers = markers.astype(np.uint8)
        
        # Find connected components for watershed seeds
        contours, _ = cv2.findContours(markers, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        markers = np.zeros(gray.shape, dtype=np.int32)
        
        # Draw contours and assign unique indices
        for i, contour in enumerate(contours):
            cv2.drawContours(markers, [contour], -1, i + 1, -1)
        
        # Apply watershed algorithm if we have markers
        if len(contours) > 0:
            cv2.watershed(image, markers)
            watershed_mask = np.zeros(gray.shape, dtype=np.uint8)
            watershed_mask[markers > 0] = 255
            final_mask = cv2.bitwise_or(final_mask, watershed_mask)
        
        # Find contours in the final mask
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on area, circularity, and convexity
        valid_contours = []
        areas = []
        
        for contour in contours:
            # Calculate contour area
            area = cv2.contourArea(contour)
            
            # Skip if area is too small or too large
            if area < self.min_area or area > self.max_area:
                continue
            
            # Calculate perimeter
            perimeter = cv2.arcLength(contour, True)
            
            # Calculate circularity: 4*pi*area/perimeter^2 (1.0 = perfect circle)
            circularity = (4 * np.pi * area) / (perimeter * perimeter + 1e-6)
            
            # Calculate convexity: area / convex hull area
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            convexity = area / (hull_area + 1e-6)
            
            # Use optimized parameters
            if circularity > 0.2 and convexity > 0.6:  # Optimized thresholds for Mpox lesions
                valid_contours.append(contour)
                areas.append(area)
        
        # Create masks for valid contours
        mask = np.zeros_like(gray)
        
        for i, contour in enumerate(valid_contours):
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

    def create_detection_image(self, image, detections, output_path=None):
        """
        Create an enhanced visualization of detected lesions.
        
        Args:
            image: Original image
            detections: Detection results from detect_lesions
            output_path: Path to save the visualization
            
        Returns:
            Visualization image
        """
        # Create a copy of the image for visualization
        vis_image = image.copy()
        
        # Create a separate overlay for the lesion masks
        overlay = np.zeros_like(image)
        
        # Draw each lesion with a unique color and proper annotation
        for i, (contour, area, box) in enumerate(zip(detections['contours'], 
                                                   detections['areas'], 
                                                   detections['boxes'])):
            # Generate a consistent color for this lesion
            color_value = (i * 35) % 180 + 40  # Avoid too dark/light colors
            color = (0, color_value, 255 - color_value)  # BGR format
            
            # Draw the contour with proper thickness
            cv2.drawContours(vis_image, [contour], -1, color, 2)
            
            # Fill the contour in the overlay
            cv2.drawContours(overlay, [contour], -1, color, -1)
            
            # Get the center of the contour for labeling
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                # If moments fail, use the box center
                x, y, x2, y2 = box
                cX = (x + x2) // 2
                cY = (y + y2) // 2
            
            # Draw lesion ID and area
            label = f"#{i+1}"
            cv2.putText(vis_image, label, (cX-15, cY), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw size information near the bounding box
            x, y, x2, y2 = box
            size_label = f"Area: {area:.0f}px"
            cv2.putText(vis_image, size_label, (x, y2 + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Blend the overlay with the visualization image
        alpha = 0.3  # Transparency factor
        cv2.addWeighted(overlay, alpha, vis_image, 1 - alpha, 0, vis_image)
        
        # Draw a summary at the top of the image
        summary = f"Lesions: {detections['lesion_count']}, Total Area: {detections['total_area']:.0f}px"
        cv2.putText(vis_image, summary, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Save visualization if output path is provided
        if output_path:
            cv2.imwrite(output_path, vis_image)
        
        return vis_image

class HeadlessLesionTracker:
    """
    Headless lesion tracker to monitor lesion progression over time.
    """
    def __init__(self, detector, data_dir=None):
        """
        Initialize the lesion tracker.

        Args:
            detector: HeadlessLesionDetector instance
            data_dir: Directory to save tracking data
        """
        self.detector = detector
        self.data_dir = data_dir

        if data_dir:
            os.makedirs(data_dir, exist_ok=True)
            self.tracking_file = os.path.join(data_dir, 'tracking_data.json')
            self.load_tracking_data()
        else:
            self.tracking_file = None
            self.patient_history = {}

    def load_tracking_data(self):
        """Load existing tracking data if available."""
        if os.path.exists(self.tracking_file):
            try:
                with open(self.tracking_file, 'r') as f:
                    self.patient_history = json.load(f)
                print(f"Loaded tracking data for {len(self.patient_history)} patients")
            except Exception as e:
                print(f"Error loading tracking data: {str(e)}")
                self.patient_history = {}
        else:
            self.patient_history = {}

    def save_tracking_data(self):
        """Save tracking data to disk."""
        if self.tracking_file:
            try:
                with open(self.tracking_file, 'w') as f:
                    json.dump(self.patient_history, f, indent=2)
                print(f"Saved tracking data for {len(self.patient_history)} patients")
            except Exception as e:
                print(f"Error saving tracking data: {str(e)}")

    def track_lesions(self, image_path, patient_id):
        """
        Detect and track lesions for a patient.

        Args:
            image_path: Path to the image file
            patient_id: Patient identifier

        Returns:
            Detection results
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Detect lesions
        detections = self.detector.detect_lesions(image)

        # Create visualization
        vis_path = None
        if self.data_dir:
            vis_dir = os.path.join(self.data_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            vis_path = os.path.join(vis_dir, f"{patient_id}_{timestamp}.png")

            self.detector.create_detection_image(image, detections, vis_path)

        # Store in patient history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record = {
            'timestamp': timestamp,
            'image_path': image_path,
            'lesion_count': detections['lesion_count'],
            'total_area': float(detections['total_area']),
            'visualization_path': vis_path
        }

        if patient_id not in self.patient_history:
            self.patient_history[patient_id] = []

        self.patient_history[patient_id].append(record)
        self.save_tracking_data()

        return detections

    def get_patient_history(self, patient_id):
        """Get the lesion detection history for a patient."""
        return self.patient_history.get(patient_id, [])

    def analyze_progression(self, patient_id):
        """
        Analyze lesion progression over time for a patient.

        Args:
            patient_id: Patient identifier

        Returns:
            Dictionary with progression analysis
        """
        history = self.patient_history.get(patient_id, [])

        if len(history) < 2:
            return None  # Not enough data

        # Sort by timestamp
        history.sort(key=lambda x: x['timestamp'])

        # Get first and last records
        first_record = history[0]
        last_record = history[-1]

        # Calculate changes
        initial_count = first_record['lesion_count']
        final_count = last_record['lesion_count']
        count_change = final_count - initial_count
        count_change_pct = (count_change / initial_count * 100) if initial_count > 0 else 0

        initial_area = first_record['total_area']
        final_area = last_record['total_area']
        area_change = final_area - initial_area
        area_change_pct = (area_change / initial_area * 100) if initial_area > 0 else 0

        # Determine status
        if count_change < 0 and area_change < 0:
            status = "Improving"
        elif count_change > 0 and area_change > 0:
            status = "Worsening"
        else:
            status = "Mixed/Stable"

        # Create visualization of progression
        chart_path = None
        if self.data_dir:
            charts_dir = os.path.join(self.data_dir, 'charts')
            os.makedirs(charts_dir, exist_ok=True)

            chart_path = os.path.join(charts_dir, f"{patient_id}_progression.png")
            self.create_progression_chart(history, chart_path)

        # Time period
        try:
            first_time = datetime.strptime(first_record['timestamp'], "%Y-%m-%d %H:%M:%S")
            last_time = datetime.strptime(last_record['timestamp'], "%Y-%m-%d %H:%M:%S")
            time_diff = last_time - first_time
            time_period = str(time_diff)
        except:
            time_period = "Unknown"

        return {
            'time_period': time_period,
            'initial_count': initial_count,
            'final_count': final_count,
            'count_change': count_change,
            'count_change_pct': count_change_pct,
            'initial_area': initial_area,
            'final_area': final_area,
            'area_change': area_change,
            'area_change_pct': area_change_pct,
            'status': status,
            'chart_path': chart_path
        }

    def create_progression_chart(self, history, output_path):
        """
        Create a chart visualizing lesion progression over time.
        Uses direct image creation instead of matplotlib.
        
        Args:
            history: List of detection records
            output_path: Path to save the chart
        """
        # Extract data
        timestamps = [record['timestamp'] for record in history]
        counts = [record['lesion_count'] for record in history]
        areas = [record['total_area'] for record in history]
        
        # Create a blank image for the chart
        chart_width, chart_height = 800, 600
        chart = np.ones((chart_height, chart_width, 3), dtype=np.uint8) * 255
        
        # Constants for chart layout
        margin = 60
        plot_width = chart_width - 2 * margin
        plot_height = (chart_height - 3 * margin) // 2
        
        # Draw chart background and borders
        cv2.rectangle(chart, (margin, margin), (margin + plot_width, margin + plot_height), 
                     (240, 240, 240), -1)
        cv2.rectangle(chart, (margin, 2 * margin + plot_height), 
                     (margin + plot_width, 2 * margin + 2 * plot_height), 
                     (240, 240, 240), -1)
        
        # Draw chart titles
        cv2.putText(chart, "Lesion Count Over Time", (margin, margin - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(chart, "Total Lesion Area Over Time", (margin, 2 * margin + plot_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Plot lesion counts
        if len(counts) > 1:
            # Find min and max for scaling
            min_count = min(counts)
            max_count = max(counts)
            if min_count == max_count:
                min_count -= 1
                max_count += 1
                
            # Calculate point coordinates
            x_points = [margin + i * plot_width // (len(counts) - 1) for i in range(len(counts))]
            y_points = [margin + plot_height - int((count - min_count) * plot_height / (max_count - min_count)) 
                       for count in counts]
            
            # Draw lines and points
            for i in range(len(counts) - 1):
                # Explicitly convert to integers for OpenCV
                pt1 = (int(x_points[i]), int(y_points[i]))
                pt2 = (int(x_points[i + 1]), int(y_points[i + 1]))
                cv2.line(chart, pt1, pt2, (0, 0, 255), 2)
                cv2.circle(chart, pt1, 5, (0, 0, 255), -1)
            
            # Explicitly convert to integers for the last point
            last_pt = (int(x_points[-1]), int(y_points[-1]))
            cv2.circle(chart, last_pt, 5, (0, 0, 255), -1)
            
            # Add count values
            for i, (x, y, count) in enumerate(zip(x_points, y_points, counts)):
                cv2.putText(chart, str(count), (int(x) - 10, int(y) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Plot lesion areas
        if len(areas) > 1:
            # Find min and max for scaling
            min_area = min(areas)
            max_area = max(areas)
            if min_area == max_area:
                min_area -= 1
                max_area += 1
                
            # Calculate point coordinates
            x_points = [margin + i * plot_width // (len(areas) - 1) for i in range(len(areas))]
            y_points = [2 * margin + 2 * plot_height - int((area - min_area) * plot_height / (max_area - min_area)) 
                       for area in areas]
            
            # Draw lines and points
            for i in range(len(areas) - 1):
                # Explicitly convert to integers for OpenCV
                pt1 = (int(x_points[i]), int(y_points[i]))
                pt2 = (int(x_points[i + 1]), int(y_points[i + 1]))
                cv2.line(chart, pt1, pt2, (0, 128, 0), 2)
                cv2.circle(chart, pt1, 5, (0, 128, 0), -1)
            
            # Explicitly convert to integers for the last point
            last_pt = (int(x_points[-1]), int(y_points[-1]))
            cv2.circle(chart, last_pt, 5, (0, 128, 0), -1)
            
            # Add area values
            for i, (x, y, area) in enumerate(zip(x_points, y_points, areas)):
                cv2.putText(chart, f"{area:.0f}", (int(x) - 20, int(y) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add timestamps at the bottom
        if len(timestamps) > 0:
            for i, (x, timestamp) in enumerate(zip(x_points, timestamps)):
                # Truncate timestamp to fit
                short_ts = timestamp.split(' ')[0] if ' ' in timestamp else timestamp
                cv2.putText(chart, short_ts, (int(x) - 40, 2 * margin + 2 * plot_height + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Save the chart
        cv2.imwrite(output_path, chart)
        
        return chart

def process_image(image_path, detector, output_dir=None):
    """
    Process a single image with the lesion detector.

    Args:
        image_path: Path to the image file
        detector: HeadlessLesionDetector instance
        output_dir: Directory to save outputs

    Returns:
        Detection results
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None

    # Detect lesions
    detections = detector.detect_mpox_lesions(image)

    # Create output directory if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create visualization
    if output_dir:
        output_filename = os.path.splitext(os.path.basename(image_path))[0] + "_detection.png"
        output_path = os.path.join(output_dir, output_filename)

        detector.create_detection_image(image, detections, output_path)

        # Print detection information
        print(f"Detected {detections['lesion_count']} lesions in {image_path}")
        print(f"Total lesion area: {detections['total_area']:.0f} pixels")
        print(f"Visualization saved to {output_path}")

    return detections

def process_directory(directory, detector, output_dir=None, extensions=('.jpg', '.jpeg', '.png')):
    """
    Process all images in a directory.

    Args:
        directory: Directory containing images
        detector: HeadlessLesionDetector instance
        output_dir: Directory to save outputs
        extensions: Tuple of valid image file extensions

    Returns:
        List of detection results
    """
    results = []

    # Get all image files
    image_files = []
    for ext in extensions:
        image_files.extend(list(Path(directory).glob(f"*{ext}")))

    # Process each image
    for image_path in image_files:
        result = process_image(str(image_path), detector, output_dir)
        if result:
            results.append({
                'image_path': str(image_path),
                'lesion_count': result['lesion_count'],
                'total_area': result['total_area']
            })

    # Create summary
    if results and output_dir:
        summary_path = os.path.join(output_dir, "detection_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Processed {len(results)} images")
        print(f"Summary saved to {summary_path}")

    return results

def find_images_recursively(directory, extensions=('.jpg', '.jpeg', '.png')):
    """
    Find all images in a directory and its subdirectories.

    Args:
        directory: Root directory to search
        extensions: Tuple of valid image file extensions

    Returns:
        List of image paths
    """
    image_files = []

    for ext in extensions:
        # Find files with matching extension in all subdirectories
        ext_files = list(Path(directory).glob(f"**/*{ext}"))
        image_files.extend(ext_files)

    return image_files

def main():
    parser = argparse.ArgumentParser(description="Headless Mpox Lesion Detector")

    # Input and output
    parser.add_argument('--input', required=True, help='Path to input image or directory')
    parser.add_argument('--output_dir', default='./outputs', help='Directory to save outputs')

    # Detector parameters
    parser.add_argument('--min_area', type=int, default=100, help='Minimum lesion area in pixels')
    parser.add_argument('--max_area', type=int, default=100000, help='Maximum lesion area in pixels')
    parser.add_argument('--threshold', choices=['otsu', 'adaptive', 'simple'], default='otsu',
                       help='Thresholding method')

    # Tracking options
    parser.add_argument('--track', action='store_true', help='Enable lesion tracking')
    parser.add_argument('--patient_id', help='Patient ID for tracking')

    # Recursive search
    parser.add_argument('--recursive', action='store_true', help='Search for images recursively')

    args = parser.parse_args()

    # Create detector
    detector = HeadlessLesionDetector(
        min_area=args.min_area,
        max_area=args.max_area,
        threshold_method=args.threshold
    )

    # Initialize tracker if tracking is enabled
    tracker = None
    if args.track and args.patient_id:
        tracker = HeadlessLesionTracker(detector, args.output_dir)
        print(f"Initialized lesion tracker for patient: {args.patient_id}")

    # Process input
    if os.path.isdir(args.input):
        # Process directory
        print(f"Processing directory: {args.input}")

        if args.recursive:
            # Find all images recursively
            image_files = find_images_recursively(args.input)
            print(f"Found {len(image_files)} images in directory and subdirectories")

            # Process each image
            results = []
            for image_path in image_files:
                result = process_image(str(image_path), detector, args.output_dir)
                if result:
                    results.append({
                        'image_path': str(image_path),
                        'lesion_count': result['lesion_count'],
                        'total_area': result['total_area']
                    })

                    # Track lesions if enabled
                    if tracker and args.patient_id:
                        tracker.track_lesions(str(image_path), args.patient_id)

            # Create summary
            if results and args.output_dir:
                summary_path = os.path.join(args.output_dir, "detection_summary.json")
                with open(summary_path, 'w') as f:
                    json.dump(results, f, indent=2)

                print(f"Processed {len(results)} images")
                print(f"Summary saved to {summary_path}")
        else:
            # Process only the top-level directory
            results = process_directory(args.input, detector, args.output_dir)

            # Track lesions if enabled
            if tracker and args.patient_id:
                for result in results:
                    tracker.track_lesions(result['image_path'], args.patient_id)

        # Analyze progression
        if tracker and args.patient_id and results:
            progression = tracker.analyze_progression(args.patient_id)
            if progression:
                print("\nProgression Analysis:")
                print(f"Time Period: {progression['time_period']}")
                print(f"Initial Count: {progression['initial_count']} → Final: {progression['final_count']} ({progression['count_change_pct']:.1f}%)")
                print(f"Initial Area: {progression['initial_area']:.0f} → Final: {progression['final_area']:.0f} ({progression['area_change_pct']:.1f}%)")
                print(f"Status: {progression['status']}")

                if progression['chart_path']:
                    print(f"Progression chart saved to: {progression['chart_path']}")
    else:
        # Process single image
        print(f"Processing image: {args.input}")

        if tracker and args.patient_id:
            # Use tracker to process image
            detections = tracker.track_lesions(args.input, args.patient_id)
            print(f"Detected {detections['lesion_count']} lesions")
            print(f"Total lesion area: {detections['total_area']:.0f} pixels")
        else:
            # Process image directly
            process_image(args.input, detector, args.output_dir)

if __name__ == "__main__":
    main()
