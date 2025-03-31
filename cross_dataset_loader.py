import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
from pathlib import Path
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2

class GenericLesionDataset(Dataset):
    """
    A flexible dataset for skin lesion segmentation that can handle different datasets
    with varying directory structures and file naming conventions.
    """
    
    def __init__(self, images_dir, masks_dir=None, transform=None, 
                 target_size=(256, 256), use_pseudo_labels=False, 
                 aug_transform=None, dataset_type="mpox", mask_suffix="_lesion"):
        """
        Args:
            images_dir (str): Directory with the images
            masks_dir (str, optional): Directory with the mask images
            transform (callable, optional): Optional transform to be applied on a sample
            target_size (tuple): Resize images to this size
            use_pseudo_labels (bool): Whether to generate pseudo labels if masks_dir is None
            aug_transform (callable, optional): Data augmentation transforms
            dataset_type (str): Type of dataset - affects file pattern matching
            mask_suffix (str): Suffix for mask files (e.g., "_lesion" for PH2)
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir) if masks_dir else None
        self.transform = transform
        self.aug_transform = aug_transform
        self.target_size = target_size
        self.use_pseudo_labels = use_pseudo_labels
        self.dataset_type = dataset_type
        self.mask_suffix = mask_suffix
        
        # Find all image files
        self.image_files = sorted([f for f in os.listdir(images_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        
        print(f"Found {len(self.image_files)} images in {images_dir}")
    
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
            # Determine mask filename based on dataset type
            if self.dataset_type == "ph2":
                # PH2 format: IMD003.bmp -> IMD003_lesion.bmp
                mask_filename = self.image_files[idx].replace('.bmp', f'{self.mask_suffix}.bmp')
            else:
                # Mpox format (assuming): image.jpg -> image_mask.png
                mask_filename = self.image_files[idx].replace('.jpg', '_mask.png').replace('.jpeg', '_mask.png')
            
            mask_path = os.path.join(self.masks_dir, mask_filename)
            
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')
                mask = mask.resize(self.target_size, Image.NEAREST)
                mask = np.array(mask) > 0  # Convert to binary
            else:
                # No mask found, create empty mask
                mask = np.zeros(self.target_size, dtype=np.uint8)
        elif self.use_pseudo_labels:
            # Generate pseudo label
            mask = self._generate_pseudo_mask(image_np)
        else:
            # No mask, create empty mask
            mask = np.zeros(self.target_size, dtype=np.uint8)
        
        # Convert boolean mask to uint8 before augmentation
        if isinstance(mask, np.ndarray) and mask.dtype == bool:
            mask = mask.astype(np.uint8) * 255
        
        # Apply data augmentation if provided
        if self.aug_transform:
            augmented = self.aug_transform(image=image_np, mask=mask)
            image_np = augmented['image']
            mask = augmented['mask']
            
            # Convert mask back to boolean after augmentation if needed
            if isinstance(mask, np.ndarray) and mask.dtype != bool:
                mask = mask > 0
        
        # Apply base transforms (normalization, etc.)
        if self.transform:
            image_np = self.transform(image_np)
        else:
            # Default normalization - check if already a tensor (from augmentation)
            if not isinstance(image_np, torch.Tensor):
                # For numpy arrays, normalize and convert to tensor
                image_np = image_np / 255.0
                image_np = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
        
        # Convert mask to tensor
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask.astype(np.float32))
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)  # Add channel dimension
        
        return {
            'image': image_np,
            'mask': mask,
            'filename': self.image_files[idx],
            'dataset': self.dataset_type
        }
    
    def _generate_pseudo_mask(self, image):
        """Generate a pseudo mask using traditional CV methods"""
        # Convert to HSV color space for better color-based segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Extract value channel (brightness)
        _, _, v_channel = cv2.split(hsv)
        
        # Create a darkness map to identify darker regions
        blurred_v = cv2.GaussianBlur(v_channel, (51, 51), 0)
        darkness_map = cv2.subtract(blurred_v, v_channel)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        darkness_enhanced = clahe.apply(darkness_map)
        
        # Threshold to identify darker regions
        _, binary = cv2.threshold(darkness_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Convert to uint8 mask
        mask = (cleaned > 0).astype(np.uint8)
        
        return mask


def create_cross_dataset_loaders(ph2_dir, mpox_dir, batch_size=16, target_size=(256, 256), num_workers=4):
    """
    Create data loaders for cross-dataset training and evaluation
    
    Args:
        ph2_dir (str): Base directory for the PH2 dataset
        mpox_dir (str): Base directory for the Mpox dataset
        batch_size (int): Batch size for data loaders
        target_size (tuple): Target image size
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Set up paths for PH2 (training)
    ph2_train_images = os.path.join(ph2_dir, 'split', 'train', 'images')
    ph2_train_masks = os.path.join(ph2_dir, 'split', 'train', 'masks')
    ph2_val_images = os.path.join(ph2_dir, 'split', 'val', 'images')
    ph2_val_masks = os.path.join(ph2_dir, 'split', 'val', 'masks')
    
    # Set up paths for Mpox (testing)
    mpox_test_images = os.path.join(mpox_dir)
    mpox_test_masks = None  # Set to None if no masks available, otherwise point to masks dir
    
    # More aggressive data augmentation for training
    train_transform = A.Compose([
        # Geometric transformations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
        
        # Color transformations (enhanced for cross-dataset transfer)
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        
        # Domain adaptation enhancements
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.ToGray(p=0.05),  # Occasionally convert to grayscale
        A.CLAHE(p=0.3),    # Contrast Limited Adaptive Histogram Equalization
        A.Posterize(p=0.1), # Reduce color information
        A.RandomGamma(p=0.3),
        
        # Elastic transform to simulate skin deformation
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
        
        # Add grid distortion and optical distortion
        A.GridDistortion(p=0.2),
        A.OpticalDistortion(distort_limit=0.2, shift_limit=0.2, p=0.2),
        
        # Quality degradation to improve robustness
        A.ImageCompression(quality_lower=80, quality_upper=100, p=0.2),
        A.GaussNoise(var_limit=(10, 50), p=0.2),
        
        # Cutouts to force model to look at different parts of the image
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.2),
        
        # Normalize and convert to tensor
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Only normalization for validation and testing
    val_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Create datasets
    train_dataset = GenericLesionDataset(
        ph2_train_images, ph2_train_masks, transform=None, target_size=target_size,
        aug_transform=train_transform, dataset_type="ph2", mask_suffix="_lesion"
    )
    
    val_dataset = GenericLesionDataset(
        ph2_val_images, ph2_val_masks, transform=None, target_size=target_size,
        aug_transform=val_transform, dataset_type="ph2", mask_suffix="_lesion"
    )
    
    # For test dataset, use Mpox images
    test_dataset = GenericLesionDataset(
        mpox_test_images, mpox_test_masks, transform=None, target_size=target_size,
        aug_transform=val_transform, dataset_type="mpox", use_pseudo_labels=(mpox_test_masks is None)
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
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
