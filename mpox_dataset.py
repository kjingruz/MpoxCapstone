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
