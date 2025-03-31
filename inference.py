import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm

# Import U-Net model architecture
from unet_model import UNet


class MpoxLesionSegmenter:
    """
    Class for segmenting Mpox lesions using trained U-Net model
    """
    def __init__(self, model_path, device=None, img_size=256):
        """
        Initialize the segmenter with a trained model
        
        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on (cpu or cuda)
            img_size: Size to resize images to before inference
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = UNet(n_channels=3, n_classes=1, bilinear=True)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Set image size
        self.img_size = img_size
        
        print(f"Model loaded successfully! (IoU: {checkpoint.get('val_iou', 'N/A'):.4f})")

    def preprocess_image(self, image):
        """
        Preprocess an image for the model
        
        Args:
            image: PIL image or numpy array
            
        Returns:
            Tensor ready for model inference
        """
        # Convert to PIL if numpy array
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Resize
        image = image.resize((self.img_size, self.img_size), Image.LANCZOS)
        
        # Convert to numpy array
        image_np = np.array(image)
        
        # Normalize
        image_np = image_np / 255.0
        image_np = (image_np - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    

    def segment_image(self, image, threshold=0.7):  # Increase threshold from 0.5 to 0.7
        """
        Segment lesions in an image with improved filtering
        
        Args:
            image: PIL image or numpy array
            threshold: Confidence threshold (0-1)
            
        Returns:
            Dictionary with mask and contours
        """
        # [existing code to get predictions]
        
        # Apply threshold
        binary_mask = (pred_np >= threshold).astype(np.uint8) * 255
        
        # Resize back to original dimensions
        binary_mask = cv2.resize(binary_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        # Apply morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and shape
        min_area = 200  # Increase from 50 to 200 pixels
        max_eccentricity = 0.95  # Filter out highly elongated shapes (likely false positives)
        filtered_contours = []
        areas = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Check area
            if area < min_area:
                continue
                
            # Check shape (filter out highly irregular shapes)
            if len(contour) >= 5:  # Ellipse fitting requires at least 5 points
                ellipse = cv2.fitEllipse(contour)
                (_, _), (major, minor), _ = ellipse
                
                # Skip highly elongated shapes (likely artifacts)
                if minor > 0 and major/minor > 5:
                    continue
                    
                # Calculate solidity (area / convex hull area)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = area / hull_area
                    # Skip shapes with very low solidity (irregular shapes)
                    if solidity < 0.4:
                        continue
            
            filtered_contours.append(contour)
            areas.append(area)
        
        return {
            'mask': binary_mask,
            'contours': filtered_contours,
            'areas': areas,
            'lesion_count': len(filtered_contours),
            'total_area': sum(areas)
        }
    
    def create_visualization(self, image, segmentation, output_path=None):
        """
        Create visualization of segmentation results
        
        Args:
            image: Original image (PIL or numpy)
            segmentation: Result from segment_image
            output_path: Path to save visualization
            
        Returns:
            Visualization image (numpy array)
        """
        # Convert to numpy if PIL
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # Create a copy
        vis_image = image.copy()
        
        # Create overlay for the mask
        overlay = np.zeros_like(image)
        
        # Draw contours with different colors
        for i, contour in enumerate(segmentation['contours']):
            # Generate color based on index
            color_value = (i * 35) % 180 + 40  # Avoid too dark/light colors
            color = (0, color_value, 255 - color_value)  # BGR format
            
            # Draw filled contour on overlay
            cv2.drawContours(overlay, [contour], -1, color, -1)
            
            # Draw contour outline on visualization
            cv2.drawContours(vis_image, [contour], -1, color, 2)
            
            # Get the center of the contour for labeling
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # Draw lesion ID
                label = f"#{i+1}"
                cv2.putText(vis_image, label, (cX-15, cY), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw area information
                area = segmentation['areas'][i]
                cv2.putText(vis_image, f"{area} px", (cX-20, cY+20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Blend overlay with visualization
        alpha = 0.3  # Transparency
        cv2.addWeighted(overlay, alpha, vis_image, 1 - alpha, 0, vis_image)
        
        # Add summary information
        summary = f"Detected: {segmentation['lesion_count']} lesions, Total area: {segmentation['total_area']} px"
        cv2.putText(vis_image, summary, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        
        return vis_image


def process_directory(model, input_dir, output_dir, extensions=('.jpg', '.jpeg', '.png')):
    """
    Process all images in a directory
    
    Args:
        model: MpoxLesionSegmenter instance
        input_dir: Directory containing images
        output_dir: Directory to save outputs
        extensions: Tuple of valid file extensions
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all image files
    image_paths = []
    for ext in extensions:
        image_paths.extend(list(Path(input_dir).glob(f"**/*{ext}")))
    
    print(f"Found {len(image_paths)} images in {input_dir}")
    
    # Process each image
    results = []
    
    for img_path in tqdm(image_paths):
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Segment image
        segmentation = model.segment_image(image)
        
        # Create visualization
        output_filename = f"{img_path.stem}_segmentation.png"
        output_path = os.path.join(output_dir, output_filename)
        
        model.create_visualization(image, segmentation, output_path)
        
        # Add to results
        results.append({
            'filename': str(img_path),
            'lesion_count': segmentation['lesion_count'],
            'total_area': segmentation['total_area']
        })
        
    # Save summary JSON
    import json
    with open(os.path.join(output_dir, 'segmentation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Processed {len(image_paths)} images. Results saved to {output_dir}")
    
    # Calculate overall statistics
    total_count = sum(r['lesion_count'] for r in results)
    total_area = sum(r['total_area'] for r in results)
    avg_count = total_count / len(results) if results else 0
    
    print(f"Total lesions detected: {total_count}")
    print(f"Average lesions per image: {avg_count:.2f}")
    print(f"Total lesion area: {total_area} pixels")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Mpox Lesion Segmentation using U-Net')
    parser.add_argument('--model', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--input', required=True, help='Input image or directory')
    parser.add_argument('--output', default='./segmentation_results', help='Output directory')
    parser.add_argument('--img_size', type=int, default=256, help='Size to resize images to')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    
    # Create model
    model = MpoxLesionSegmenter(args.model, device, args.img_size)
    
    # Check if input is a directory or a file
    if os.path.isdir(args.input):
        process_directory(model, args.input, args.output)
    else:
        # Single image processing
        image = Image.open(args.input).convert("RGB")
        
        # Segment image
        segmentation = model.segment_image(image, args.threshold)
        
        # Create output directory
        os.makedirs(args.output, exist_ok=True)
        
        # Create visualization
        output_filename = os.path.basename(args.input).split('.')[0] + "_segmentation.png"
        output_path = os.path.join(args.output, output_filename)
        
        vis = model.create_visualization(image, segmentation, output_path)
        
        # Print results
        print(f"Detected {segmentation['lesion_count']} lesions")
        print(f"Total area: {segmentation['total_area']} pixels")
        print(f"Visualization saved to {output_path}")
        
        # Display visualization if running in a notebook
        try:
            from IPython.display import display
            plt.figure(figsize=(10, 8))
            plt.imshow(vis)
            plt.axis('off')
            plt.title(f"Detected {segmentation['lesion_count']} lesions")
            plt.show()
        except ImportError:
            pass


if __name__ == "__main__":
    main()
