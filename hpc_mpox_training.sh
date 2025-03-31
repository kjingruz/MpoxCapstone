#!/bin/bash

###############################################################################
# Slurm Job Configuration
###############################################################################
#SBATCH --job-name=Mpox_UNet
#SBATCH --output=Mpox_UNet_%j.log
#SBATCH --error=Mpox_UNet_%j.log
#SBATCH --time=12:00:00

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=main

###############################################################################
# 1) HPC Modules
###############################################################################
module load python/3.10
module load cuda/11.8.0
module load cudnn/8.6.0

###############################################################################
# 2) Virtual Env & Directories
###############################################################################
VENV_DIR="$HOME/mpox_venv"
SCRIPT_DIR="$(pwd)"
WEIGHTS_DIR="$SCRIPT_DIR/model_weights"
RESULTS_DIR="$SCRIPT_DIR/results"
VISUALIZATION_DIR="$SCRIPT_DIR/visualizations"

# Create necessary directories
mkdir -p "$WEIGHTS_DIR"
mkdir -p "$RESULTS_DIR"
mkdir -p "$VISUALIZATION_DIR"

echo "===== GPU status before job starts ====="
nvidia-smi
echo "========================================="

###############################################################################
# 3) Virtual Env Setup
###############################################################################
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating new virtual environment in $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel

###############################################################################
# 4) Install PyTorch and Dependencies
###############################################################################
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas matplotlib scikit-learn tqdm opencv-python Pillow albumentations

###############################################################################
# 5) Configuration Parameters
###############################################################################
SEED=42
MAX_EPOCHS=50
BATCH_SIZE=16
LEARNING_RATE=0.001
IMAGE_SIZE=256
USE_PSEUDO_LABELS=true

###############################################################################
# 6) Copy Python Files to Working Directory
###############################################################################
echo "Setting up Python scripts..."

# Create U-Net model file
cat > "$SCRIPT_DIR/unet_model.py" << 'EOL'
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2D => BN => ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
EOL

# Create Dataset file
cat > "$SCRIPT_DIR/mpox_dataset.py" << 'EOL'
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
from pathlib import Path
import random


class MpoxDataset(Dataset):
    """Mpox lesion segmentation dataset"""
    
    def __init__(self, images_dir, masks_dir=None, transform=None, target_size=(256, 256), 
                 use_pseudo_labels=False, aug_transform=None):
        """
        Args:
            images_dir (str): Directory with the mpox images
            masks_dir (str, optional): Directory with the mask images. If None and use_pseudo_labels is True,
                                      pseudo labels will be generated.
            transform (callable, optional): Optional transform to be applied on a sample
            target_size (tuple): Resize images to this size
            use_pseudo_labels (bool): Whether to generate pseudo labels if masks_dir is None
            aug_transform (callable, optional): Data augmentation transforms
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir) if masks_dir else None
        self.transform = transform
        self.aug_transform = aug_transform
        self.target_size = target_size
        self.use_pseudo_labels = use_pseudo_labels
        
        # Find all image files
        self.image_files = sorted([f for f in os.listdir(images_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Get image path
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Resize to target size
        image = image.resize(self.target_size, Image.LANCZOS)
        
        # Convert to numpy for processing if needed
        image_np = np.array(image)
        
        # Load or generate mask
        if self.masks_dir:
            # Try to find corresponding mask
            mask_filename = self.image_files[idx].replace('.jpg', '_mask.png').replace('.jpeg', '_mask.png')
            mask_path = os.path.join(self.masks_dir, mask_filename)
            
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')
                mask = mask.resize(self.target_size, Image.NEAREST)
                mask = np.array(mask) > 0  # Convert to binary
            else:
                # No mask found, create empty mask
                mask = np.zeros(self.target_size, dtype=np.bool_)
        elif self.use_pseudo_labels:
            # Generate pseudo label using our previous detection approach
            mask = self._generate_pseudo_mask(image_np)
        else:
            # No mask, create empty mask
            mask = np.zeros(self.target_size, dtype=np.bool_)
        
        # Apply data augmentation if provided
        if self.aug_transform:
            augmented = self.aug_transform(image=image_np, mask=mask)
            image_np = augmented['image']
            mask = augmented['mask']
        
        # Apply base transforms (normalization, etc.)
        if self.transform:
            image_np = self.transform(image_np)
        else:
            # Default normalization
            image_np = image_np / 255.0
            image_np = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
        
        # Convert mask to tensor
        mask = torch.from_numpy(mask.astype(np.float32))
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)  # Add channel dimension
        
        return {
            'image': image_np,
            'mask': mask,
            'filename': self.image_files[idx]
        }
    
    def _generate_pseudo_mask(self, image):
        """Generate a pseudo mask using traditional CV methods"""
        # Convert to HSV color space for better color-based segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Extract value channel (brightness)
        _, _, v_channel = cv2.split(hsv)
        
        # Create a darkness map to identify darker regions
        # First blur to get average skin tone
        blurred_v = cv2.GaussianBlur(v_channel, (51, 51), 0)
        # Subtract to find regions darker than surroundings
        darkness_map = cv2.subtract(blurred_v, v_channel)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        darkness_enhanced = clahe.apply(darkness_map)
        
        # Threshold to identify darker regions
        _, binary = cv2.threshold(darkness_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Convert to boolean mask
        mask = cleaned > 0
        
        return mask


def get_data_loaders(images_dir, masks_dir=None, batch_size=8, val_split=0.2, 
                    use_pseudo_labels=False, target_size=(256, 256), num_workers=4):
    """
    Create train and validation data loaders
    
    Args:
        images_dir (str): Directory with images
        masks_dir (str, optional): Directory with mask images
        batch_size (int): Batch size for data loaders
        val_split (float): Validation split ratio (0-1)
        use_pseudo_labels (bool): Whether to use pseudo labels if no masks provided
        target_size (tuple): Target image size
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Create list of image files
    all_images = sorted([f for f in os.listdir(images_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # Split into train and validation
    indices = list(range(len(all_images)))
    random.seed(42)  # For reproducibility
    random.shuffle(indices)
    
    val_size = int(len(indices) * val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    train_images = [all_images[i] for i in train_indices]
    val_images = [all_images[i] for i in val_indices]
    
    # Create train and validation directories
    train_dir = os.path.join(images_dir, '../train')
    val_dir = os.path.join(images_dir, '../val')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Create symlinks to original images
    for img in train_images:
        src = os.path.join(images_dir, img)
        dst = os.path.join(train_dir, img)
        if not os.path.exists(dst):
            if os.name == 'nt':  # Windows
                import shutil
                shutil.copy2(src, dst)
            else:  # Unix
                os.symlink(src, dst)
    
    for img in val_images:
        src = os.path.join(images_dir, img)
        dst = os.path.join(val_dir, img)
        if not os.path.exists(dst):
            if os.name == 'nt':  # Windows
                import shutil
                shutil.copy2(src, dst)
            else:  # Unix
                os.symlink(src, dst)
    
    # Create datasets
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        # Data augmentation for training
        train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussianBlur(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Only normalization for validation
        val_transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        train_dataset = MpoxDataset(
            train_dir, masks_dir, transform=None,
            target_size=target_size, use_pseudo_labels=use_pseudo_labels,
            aug_transform=train_transform
        )
        
        val_dataset = MpoxDataset(
            val_dir, masks_dir, transform=None,
            target_size=target_size, use_pseudo_labels=use_pseudo_labels,
            aug_transform=val_transform
        )
    except ImportError:
        # Fallback if albumentations is not available
        print("Albumentations not available, using basic transforms")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        train_dataset = MpoxDataset(
            train_dir, masks_dir, transform=transform,
            target_size=target_size, use_pseudo_labels=use_pseudo_labels
        )
        
        val_dataset = MpoxDataset(
            val_dir, masks_dir, transform=transform,
            target_size=target_size, use_pseudo_labels=use_pseudo_labels
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader
EOL

# Create Training script
cat > "$SCRIPT_DIR/train_unet.py" << 'EOL'
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import json

# Import our model and dataset
from unet_model import UNet
from mpox_dataset import get_data_loaders, MpoxDataset

# Dice loss for better segmentation performance
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, logits, targets):
        # Apply sigmoid to get 0-1 probabilities
        probs = torch.sigmoid(logits)
        
        # Flatten
        batch_size = probs.shape[0]
        probs = probs.view(batch_size, -1)
        targets = targets.view(batch_size, -1)
        
        # Compute dice score
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        
        # Calculate Dice score
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return negative dice (we want to maximize dice, minimize loss)
        return 1.0 - dice.mean()

# Combined loss for better performance
class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        
    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

# Function to calculate IoU (Intersection over Union) metric
def iou_score(preds, targets):
    # Apply sigmoid and threshold to get binary predictions
    preds = (torch.sigmoid(preds) > 0.5).float()
    
    # Flatten tensors
    batch_size = preds.shape[0]
    preds = preds.view(batch_size, -1)
    targets = targets.view(batch_size, -1)
    
    # Calculate intersection and union
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection
    
    # Add small constant to avoid division by zero
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    return iou.mean()

# Training function for one epoch
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    epoch_iou = 0
    
    # Wrap dataloader with tqdm for progress bar
    with tqdm(dataloader, unit="batch") as tepoch:
        for batch in tepoch:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, masks)
            
            # Calculate IoU
            batch_iou = iou_score(outputs, masks)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            epoch_iou += batch_iou.item()
            
            # Update progress bar
            tepoch.set_postfix(loss=loss.item(), iou=batch_iou.item())
    
    # Return average loss and IoU for the epoch
    return epoch_loss / len(dataloader), epoch_iou / len(dataloader)

# Validation function
def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0
    val_iou = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, masks)
            
            # Calculate IoU
            batch_iou = iou_score(outputs, masks)
            
            # Update metrics
            val_loss += loss.item()
            val_iou += batch_iou.item()
    
    # Return average loss and IoU
    return val_loss / len(dataloader), val_iou / len(dataloader)

# Visualization function
def visualize_predictions(model, dataloader, device, num_samples=3, output_dir=None):
    model.eval()
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
                
            images = batch['image'].to(device)
            masks = batch['mask']
            filenames = batch['filename']
            
            # Forward pass
            outputs = model(images)
            
            # Apply sigmoid and threshold
            preds = torch.sigmoid(outputs) > 0.5
            
            # Loop through batch
            for j in range(images.shape[0]):
                # Convert tensors to numpy for visualization
                image = images[j].cpu().numpy()
                mask = masks[j].cpu().numpy()
                pred = preds[j].cpu().numpy()
                
                # Transpose from CHW to HWC for visualization
                image = np.transpose(image, (1, 2, 0))
                
                # Denormalize image
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image = std * image + mean
                image = np.clip(image, 0, 1)
                
                # Create figure
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Plot original image
                axes[0].imshow(image)
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                # Plot ground truth mask
                axes[1].imshow(mask[0], cmap='gray')
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')
                
                # Plot prediction
                axes[2].imshow(pred[0], cmap='gray')
                axes[2].set_title('Prediction')
                axes[2].axis('off')
                
                plt.tight_layout()
                
                # Save figure if output_dir provided
                if output_dir:
                    plt.savefig(os.path.join(output_dir, f"{filenames[j]}_pred.png"))
                
                plt.close()
                
                # Limit total samples
                if i * images.shape[0] + j + 1 >= num_samples:
                    break

# Function to run inference on test data and save results
def run_inference(model, test_dir, output_dir, device, target_size=(256, 256)):
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a dataset for test data
    test_dataset = MpoxDataset(
        test_dir, masks_dir=None, 
        target_size=target_size, use_pseudo_labels=False
    )
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, 
        num_workers=1, pin_memory=True
    )
    
    # Set model to evaluation mode
    model.eval()
    
    # Create a list to store results
    results = []
    
    # Process each image
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Running inference"):
            # Get the image
            image = batch['image'].to(device)
            filename = batch['filename'][0]
            
            # Run inference
            output = model(image)
            pred_mask = (torch.sigmoid(output) > 0.5).float()
            
            # Convert to numpy
            pred_numpy = pred_mask[0, 0].cpu().numpy()
            
            # Find contours
            contours, _ = cv2.findContours(
                (pred_numpy * 255).astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Calculate areas
            areas = [cv2.contourArea(contour) for contour in contours]
            
            # Create visualization
            # Load and denormalize the image
            image_np = image[0].cpu().numpy()
            image_np = np.transpose(image_np, (1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_np = std * image_np + mean
            image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
            
            # Convert to BGR for OpenCV
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            # Create a copy for drawing
            vis_image = image_bgr.copy()
            
            # Draw contours
            for i, contour in enumerate(contours):
                color = (0, 0, 255)  # Red color for contours
                cv2.drawContours(vis_image, [contour], -1, color, 2)
                
                # Get centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Add label
                    label = f"#{i+1}"
                    cv2.putText(vis_image, label, (cx-10, cy), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Add summary at the top
            summary = f"Lesions: {len(contours)}, Total Area: {sum(areas):.0f}px"
            cv2.putText(vis_image, summary, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Save visualization
            cv2.imwrite(os.path.join(output_dir, f"{filename}_segmentation.png"), vis_image)
            
            # Save mask
            cv2.imwrite(os.path.join(output_dir, f"{filename}_mask.png"), 
                        (pred_numpy * 255).astype(np.uint8))
            
            # Add to results
            results.append({
                'filename': filename,
                'lesion_count': len(contours),
                'total_area': float(sum(areas)),
                'individual_areas': [float(area) for area in areas]
            })
    
    # Save results to JSON
    with open(os.path.join(output_dir, 'inference_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Inference completed. Results saved to {output_dir}")
    return results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train U-Net for Mpox Lesion Segmentation')
    parser.add_argument('--images_dir', required=True, help='Directory containing images')
    parser.add_argument('--masks_dir', help='Directory containing mask annotations (optional)')
    parser.add_argument('--output_dir', default='./outputs', help='Output directory for models and results')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=256, help='Image size (square)')
    parser.add_argument('--use_pseudo_labels', action='store_true', help='Generate pseudo labels if no masks')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--test_dir', help='Directory with test images for inference after training')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    device = torch.device("cpu")
    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    # Log arguments
    with open(os.path.join(model_dir, "args.txt"), "w") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    # Create data loaders
    train_loader, val_loader = get_data_loaders(
        args.images_dir,
        args.masks_dir,
        batch_size=args.batch_size,
        val_split=0.2,
        use_pseudo_labels=args.use_pseudo_labels,
        target_size=(args.img_size, args.img_size),
        num_workers=args.num_workers
    )
    
    print(f"Training with {len(train_loader.dataset)} images")
    print(f"Validating with {len(val_loader.dataset)} images")
    
    # Create model
    model = UNet(n_channels=3, n_classes=1, bilinear=True)
    model.to(device)
    
    # Create optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    
    # Training loop
    best_val_iou = 0.0
    patience = 10
    patience_counter = 0
    history = {
        'train_loss': [],
        'train_iou': [],
        'val_loss': [],
        'val_iou': []
    }
    
    print("Starting training...")
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train for one epoch
        train_loss, train_iou = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_iou = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
        
        # Save checkpoint if this is the best model so far
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            patience_counter = 0
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_iou,
                'history': history
            }
            
            checkpoint_path = os.path.join(model_dir, "best_model.pth")
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved best model checkpoint (IoU: {val_iou:.4f})")
        else:
            patience_counter += 1
            
        # Save intermediate checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_iou,
                'history': history
            }
            
            checkpoint_path = os.path.join(model_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(checkpoint, checkpoint_path)
            
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save final model
    final_checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_iou': val_iou,
        'history': history
    }
    
    final_checkpoint_path = os.path.join(model_dir, "final_model.pth")
    torch.save(final_checkpoint, final_checkpoint_path)
    print(f"Saved final model checkpoint")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_iou'], label='Train')
    plt.plot(history['val_iou'], label='Validation')
    plt.title('IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "training_history.png"))
    plt.close()
    
    # Visualize some predictions
    print("Generating prediction visualizations...")
    vis_dir = os.path.join(model_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    visualize_predictions(model, val_loader, device, num_samples=10, output_dir=vis_dir)
    
    # Run inference on test data if provided
    if args.test_dir:
        print(f"Running inference on test data in {args.test_dir}...")
        inference_dir = os.path.join(model_dir, "inference_results")
        run_inference(model, args.test_dir, inference_dir, device, target_size=(args.img_size, args.img_size))
    
    print(f"Training completed. Results saved to {model_dir}")
    
    # Copy the best model to a standardized location
    best_model_path = os.path.join(model_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        standardized_path = os.path.join(args.output_dir, "best_unet_model.pth")
        shutil.copy(best_model_path, standardized_path)
        print(f"Copied best model to {standardized_path}")
    
    return best_model_path, best_val_iou


if __name__ == "__main__":
    import cv2
    import shutil
    main()
EOL

# Create inference script
cat > "$SCRIPT_DIR/inference.py" << 'EOL'
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
import json

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
    
    def segment_image(self, image, threshold=0.5):
        """
        Segment lesions in an image
        
        Args:
            image: PIL image or numpy array
            threshold: Confidence threshold (0-1)
            
        Returns:
            Dictionary with mask and contours
        """
        # Get original image dimensions
        if isinstance(image, np.ndarray):
            orig_h, orig_w = image.shape[:2]
        else:
            orig_w, orig_h = image.size
        
        # Preprocess image
        image_tensor = self.preprocess_image(image)
        
        # Move to device
        image_tensor = image_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(image_tensor)
            pred = torch.sigmoid(output)
        
        # Convert to numpy
        pred_np = pred.squeeze().cpu().numpy()
        
        # Apply threshold
        binary_mask = (pred_np >= threshold).astype(np.uint8) * 255
        
        # Resize back to original dimensions
        binary_mask = cv2.resize(binary_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        min_area = 50  # Minimum area in pixels
        filtered_contours = []
        areas = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
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


if __name__ == "__main__":
    main()
EOL

###############################################################################
# 7) Training
###############################################################################
echo "Starting U-Net training..."

# Run training with specified parameters
python train_unet.py \
    --images_dir "$SCRIPT_DIR/data/Monkey_Pox" \
    --output_dir "$WEIGHTS_DIR" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$MAX_EPOCHS" \
    --lr "$LEARNING_RATE" \
    --img_size "$IMAGE_SIZE" \
    --num_workers 8 \
    `if [ "$USE_PSEUDO_LABELS" = true ]; then echo "--use_pseudo_labels"; fi`

# Find the best model
BEST_MODEL=$(find "$WEIGHTS_DIR" -name "best_model.pth" | head -1)

if [ -z "$BEST_MODEL" ]; then
    echo "ERROR: No best model found! Checking for any checkpoint..."
    BEST_MODEL=$(find "$WEIGHTS_DIR" -name "*.pth" | head -1)
    
    if [ -z "$BEST_MODEL" ]; then
        echo "ERROR: No model checkpoints found. Training may have failed."
        exit 1
    fi
fi

echo "Best model found: $BEST_MODEL"

###############################################################################
# 8) Inference
###############################################################################
echo "Running inference on Mpox data..."

# Run inference on the Mpox dataset
python inference.py \
    --model "$BEST_MODEL" \
    --input "$SCRIPT_DIR/data/Monkey_Pox" \
    --output "$RESULTS_DIR" \
    --img_size "$IMAGE_SIZE"

echo "============================================================="
echo "JOB COMPLETED SUCCESSFULLY"
echo "============================================================="
echo "Results saved to: $RESULTS_DIR"
echo "Best model saved to: $BEST_MODEL"
echo "============================================================="

# Display GPU status after job completion
echo "===== GPU status after job completion ====="
nvidia-smi
echo "=========================================="
