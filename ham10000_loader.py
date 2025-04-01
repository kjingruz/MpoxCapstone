import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import random

class HAM10000Dataset(Dataset):
    """Dataset class for HAM10000 skin lesion dataset"""
    
    def __init__(self, images_dir, transform=None, target_size=(256, 256)):
        """
        Args:
            images_dir (str): Directory with HAM10000 images (ISIC_*.jpg)
            transform (callable, optional): Transform to apply to the images
            target_size (tuple): Size to resize images to
        """
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.target_size = target_size
        
        # Find all image files
        self.image_files = sorted([f for f in os.listdir(images_dir) 
                                  if f.startswith('ISIC_') and f.lower().endswith('.jpg')])
        
        print(f"Found {len(self.image_files)} images in HAM10000 dataset")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Get image path
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Resize to target size
        image = image.resize(self.target_size, Image.LANCZOS)
        
        # Convert to numpy for processing
        image_np = np.array(image)
        
        # Apply transform if provided
        if self.transform:
            augmented = self.transform(image=image_np)
            image_np = augmented['image']
        
        return {
            'image': image_np,
            'filename': self.image_files[idx]
        }


def create_ham10000_loader(images_dir, batch_size=32, target_size=(256, 256), num_workers=4):
    """Create a data loader for HAM10000 dataset for pretraining
    
    Args:
        images_dir (str): Directory containing HAM10000 images
        batch_size (int): Batch size
        target_size (tuple): Target size for images
        num_workers (int): Number of workers for data loading
        
    Returns:
        DataLoader: PyTorch data loader for HAM10000
    """
    # Create strong augmentations for self-supervised learning
    transform = A.Compose([
        # Use size parameter instead of height/width for compatibility
        A.RandomResizedCrop(size=target_size, scale=(0.2, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.ToGray(p=0.2),
        A.GaussNoise(p=0.2),
        # Skin lesion specific augmentations
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.3),
        A.GridDistortion(p=0.2),
        A.OpticalDistortion(distort_limit=0.2, shift_limit=0.05, p=0.2),
        # Normalize and convert to tensor
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Create dataset
    dataset = HAM10000Dataset(
        images_dir=images_dir,
        transform=transform,
        target_size=target_size
    )
    
    # Create and return data loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader


# Contrastive learning dataset variant for HAM10000
class HAM10000ContrastiveDataset(Dataset):
    """Dataset for contrastive learning using HAM10000"""
    
    def __init__(self, images_dir, transform=None, target_size=(256, 256)):
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.target_size = target_size
        
        # Find all image files
        self.image_files = sorted([f for f in os.listdir(images_dir) 
                                  if f.startswith('ISIC_') and f.lower().endswith('.jpg')])
        
        print(f"Found {len(self.image_files)} images for contrastive learning")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        # Resize to target size
        image = image.resize(self.target_size, Image.LANCZOS)
        
        # Convert to numpy array
        image_np = np.array(image)
        
        # Create two different augmentations of the same image
        if self.transform:
            augmented1 = self.transform(image=image_np)
            augmented2 = self.transform(image=image_np)
            
            return {
                'view1': augmented1['image'],
                'view2': augmented2['image'],
                'filename': self.image_files[idx]
            }
        else:
            # If no transform, convert to tensor
            image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float() / 255.0
            return {
                'view1': image_tensor,
                'view2': image_tensor,
                'filename': self.image_files[idx]
            }


def create_ham10000_contrastive_loader(images_dir, batch_size=32, target_size=(256, 256), num_workers=4):
    """Create a data loader for HAM10000 dataset specifically for contrastive learning
    
    Args:
        images_dir (str): Directory containing HAM10000 images
        batch_size (int): Batch size
        target_size (tuple): Target size for images
        num_workers (int): Number of workers for data loading
        
    Returns:
        DataLoader: PyTorch data loader for contrastive learning
    """
    # Create strong augmentations for contrastive learning
    transform = A.Compose([
        # Use size parameter instead of height/width for compatibility
        A.RandomResizedCrop(size=target_size, scale=(0.2, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.ToGray(p=0.2),
        A.GaussNoise(var_limit=(10, 50), p=0.2),
        # Skin lesion specific augmentations
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.3),
        # Normalize and convert to tensor
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Create dataset
    dataset = HAM10000ContrastiveDataset(
        images_dir=images_dir,
        transform=transform,
        target_size=target_size
    )
    
    # Create and return data loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader
