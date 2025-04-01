import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import json

# Import our models and dataset handling
from unet_model import UNet
try:
    from attention_unet import AttentionUNet
except ImportError:
    print("AttentionUNet not found, will use standard UNet instead")
    AttentionUNet = None

from unet_encoder import UNetEncoder
from cross_dataset_loader import GenericLesionDataset

# Loss functions and metrics
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, logits, targets):
        # Apply sigmoid to get 0-1 probabilities
        probs = torch.sigmoid(logits)
        
        # Make sure both are float tensors
        probs = probs.float()
        targets = targets.float()
        
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

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = 1e-6
        
    def forward(self, inputs, targets):
        # Ensure targets has the proper shape and type
        if len(targets.shape) == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        
        # Get probabilities
        probs = torch.sigmoid(inputs)
        
        # Calculate focal weights
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weights = (1 - pt) ** self.gamma
        
        # Apply weights to BCE loss
        bce = -targets * torch.log(probs + self.eps) - (1 - targets) * torch.log(1 - probs + self.eps)
        loss = self.alpha * focal_weights * bce
        
        return loss.mean()

# Enhanced combined loss with regularization
class EnhancedCombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.3, dice_weight=0.5, focal_weight=0.2, reg_weight=0.001):
        super(EnhancedCombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.reg_weight = reg_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha=0.8, gamma=2.0)
        
    def forward(self, logits, targets, model=None):
        # Ensure targets has the same shape as logits
        if len(targets.shape) == 3:
            targets = targets.unsqueeze(1)
        
        # Ensure float type (critical for loss computation)
        targets = targets.float()
        
        # Calculate individual losses
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        focal_loss = self.focal(logits, targets)
        
        # L2 regularization if model is provided
        reg_loss = 0.0
        if model is not None and self.reg_weight > 0:
            for param in model.parameters():
                reg_loss += torch.norm(param, 2)
        
        # Combined loss
        return (self.bce_weight * bce_loss + 
                self.dice_weight * dice_loss + 
                self.focal_weight * focal_loss + 
                self.reg_weight * reg_loss)

# Function to calculate IoU (Intersection over Union) metric
def iou_score(preds, targets):
    # Apply sigmoid and threshold to get binary predictions
    preds = (torch.sigmoid(preds) > 0.5).float()
    
    # Ensure targets has channel dimension for consistent shape
    if len(targets.shape) == 3:
        targets = targets.unsqueeze(1)
    
    # Ensure targets is a float tensor for consistent calculations
    targets = targets.float()
    
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

# Clinical metrics calculation function
def evaluate_clinical_metrics(model, dataloader, threshold=0.7, device='cuda', is_distributed=False, rank=0):
    """Calculate simple metrics without connected components"""
    model.eval()
    
    # Use pixel-based metrics instead of component-based for now
    total_pixels = 0
    total_tp_pixels = 0
    total_fp_pixels = 0
    total_fn_pixels = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Ensure masks have channel dimension and correct type
            if len(masks.shape) == 3:
                masks = masks.unsqueeze(1)
            
            # Ensure float type
            masks = masks.float()
            
            # Forward pass
            outputs = model(images)
            
            # Convert to binary predictions
            preds = (torch.sigmoid(outputs) > threshold).float()
            
            # Convert to numpy and calculate pixel-wise metrics
            for i in range(images.shape[0]):
                pred = preds[i, 0].cpu().numpy()
                mask = masks[i, 0].cpu().numpy()
                
                # Convert to binary
                pred_binary = pred > 0
                mask_binary = mask > 0
                
                # Calculate metrics
                tp = np.logical_and(pred_binary, mask_binary).sum()
                fp = np.logical_and(pred_binary, ~mask_binary).sum()
                fn = np.logical_and(~pred_binary, mask_binary).sum()
                
                total_tp_pixels += tp
                total_fp_pixels += fp
                total_fn_pixels += fn
                total_pixels += pred.size
    
    # Handle metrics calculation for distributed setup
    if is_distributed:
        # Convert counters to tensors
        tp_tensor = torch.tensor([total_tp_pixels], device=device)
        fp_tensor = torch.tensor([total_fp_pixels], device=device)
        fn_tensor = torch.tensor([total_fn_pixels], device=device)
        total_tensor = torch.tensor([total_pixels], device=device)
        
        # Sum across all processes
        dist.all_reduce(tp_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(fp_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(fn_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        
        # Convert back to numpy
        total_tp_pixels = tp_tensor.item()
        total_fp_pixels = fp_tensor.item()
        total_fn_pixels = fn_tensor.item()
        total_pixels = total_tensor.item()
    
    # Calculate metrics
    precision = total_tp_pixels / (total_tp_pixels + total_fp_pixels) if (total_tp_pixels + total_fp_pixels) > 0 else 0
    recall = total_tp_pixels / (total_tp_pixels + total_fn_pixels) if (total_tp_pixels + total_fn_pixels) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Estimate average lesions based on reasonable assumptions
    # Assuming average lesion is about 500 pixels
    est_lesion_size = 500
    est_true_lesions = total_tp_pixels / est_lesion_size if est_lesion_size > 0 else 0
    est_pred_lesions = (total_tp_pixels + total_fp_pixels) / est_lesion_size if est_lesion_size > 0 else 0
    
    if rank == 0:  # Only print from the main process
        print(f"Pixel-based metrics: TP={total_tp_pixels}, FP={total_fp_pixels}, FN={total_fn_pixels}")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_lesions_true': est_true_lesions / len(dataloader.dataset),
        'avg_lesions_pred': est_pred_lesions / len(dataloader.dataset),
        'false_positive_rate': total_fp_pixels / total_pixels if total_pixels > 0 else 0
    }

# Training function for one epoch
def train_epoch(model, dataloader, optimizer, criterion, device, rank, accumulation_steps=1):
    model.train()
    epoch_loss = 0
    epoch_iou = 0
    
    # Wrap dataloader with tqdm for progress bar (only on rank 0)
    if rank == 0:
        pbar = tqdm(dataloader, unit="batch")
    else:
        pbar = dataloader
    
    # Zero gradients before starting
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Ensure masks have channel dimension
        if len(masks.shape) == 3:
            masks = masks.unsqueeze(1)
        
        # Ensure float type
        masks = masks.float()
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs, masks, model)
        
        # Normalize loss by accumulation steps
        loss = loss / accumulation_steps
        
        # Calculate IoU
        batch_iou = iou_score(outputs, masks)
        
        # Backward pass
        loss.backward()
        
        # Update weights only after accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0 or batch_idx == len(dataloader) - 1:
            optimizer.step()
            optimizer.zero_grad()
        
        # Update metrics (use non-normalized loss for reporting)
        epoch_loss += loss.item() * accumulation_steps
        epoch_iou += batch_iou.item()
        
        # Update progress bar (only on rank 0)
        if rank == 0:
            pbar.set_postfix(loss=loss.item() * accumulation_steps, iou=batch_iou.item())
    
    # Calculate average metrics
    avg_loss = epoch_loss / len(dataloader)
    avg_iou = epoch_iou / len(dataloader)
    
    # Average losses across all processes
    if dist.is_initialized():
        avg_loss_tensor = torch.tensor([avg_loss], device=device)
        avg_iou_tensor = torch.tensor([avg_iou], device=device)
        
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(avg_iou_tensor, op=dist.ReduceOp.SUM)
        
        avg_loss = avg_loss_tensor.item() / dist.get_world_size()
        avg_iou = avg_iou_tensor.item() / dist.get_world_size()
    
    return avg_loss, avg_iou

# Validation function
def validate(model, dataloader, criterion, device, rank):
    model.eval()
    val_loss = 0
    val_iou = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Ensure masks have channel dimension
            if len(masks.shape) == 3:
                masks = masks.unsqueeze(1)
            
            # Ensure float type
            masks = masks.float()
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss - pass model for regularization, same as in training
            loss = criterion(outputs, masks, model)
            
            # Calculate IoU
            batch_iou = iou_score(outputs, masks)
            
            # Update metrics
            val_loss += loss.item()
            val_iou += batch_iou.item()
    
    # Calculate average metrics
    avg_loss = val_loss / len(dataloader)
    avg_iou = val_iou / len(dataloader)
    
    # Average losses across all processes
    if dist.is_initialized():
        avg_loss_tensor = torch.tensor([avg_loss], device=device)
        avg_iou_tensor = torch.tensor([avg_iou], device=device)
        
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(avg_iou_tensor, op=dist.ReduceOp.SUM)
        
        avg_loss = avg_loss_tensor.item() / dist.get_world_size()
        avg_iou = avg_iou_tensor.item() / dist.get_world_size()
    
    return avg_loss, avg_iou

# Visualization function
def visualize_predictions(model, dataloader, device, num_samples=3, output_dir=None, rank=0):
    """
    Visualize model predictions (only run on rank 0)
    """
    if rank != 0:
        return
        
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
                
                # Ensure mask has the right shape for visualization
                if len(masks.shape) == 4:
                    mask = masks[j, 0].cpu().numpy()  # Take channel 0
                else:
                    mask = masks[j].cpu().numpy()
                
                pred = preds[j, 0].cpu().numpy()  # Take channel 0
                
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
                if len(mask.shape) == 3:
                    axes[1].imshow(mask[0], cmap='gray')
                else:
                    axes[1].imshow(mask, cmap='gray')
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')
                
                # Plot prediction
                axes[2].imshow(pred, cmap='gray')
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

# Setup distributed training
def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12350'  # Different port from pretraining
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl',  # Use NCCL backend for GPU training
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    # Set device for current process
    torch.cuda.set_device(rank)

# Clean up distributed resources
def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

# Create distributed data samplers and loaders
def create_distributed_dataloader(dataset, batch_size, num_workers, world_size, rank, shuffle=True):
    # Create a distributed sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle
    )
    
    # Create data loader with sampler
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader, sampler

# Create datasets for cross-dataset evaluation
def create_datasets(ph2_dir, mpox_dir, target_size=(256, 256)):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    # Set up paths for PH2 (training)
    ph2_train_images = os.path.join(ph2_dir, 'split', 'train', 'images')
    ph2_train_masks = os.path.join(ph2_dir, 'split', 'train', 'masks')
    ph2_val_images = os.path.join(ph2_dir, 'split', 'val', 'images')
    ph2_val_masks = os.path.join(ph2_dir, 'split', 'val', 'masks')
    
    # Set up paths for Mpox (testing)
    mpox_test_images = os.path.join(mpox_dir)
    mpox_test_masks = None  # Set to None if no masks available, otherwise point to masks dir
    
    # Data augmentation for training
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
        
        # Elastic transform to simulate skin deformation
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
        
        # Add grid distortion and optical distortion
        A.GridDistortion(p=0.2),
        A.OpticalDistortion(distort_limit=0.2, shift_limit=0.2, p=0.2),
        
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
    
    return train_dataset, val_dataset, test_dataset

class TransferLearningUNet(nn.Module):
    """
    UNet with pretrained encoder for transfer learning
    """
    def __init__(self, base_model, pretrained_encoder, freeze_encoder=True):
        super(TransferLearningUNet, self).__init__()
        
        # Replace base model's encoder with pretrained encoder
        self.inc = pretrained_encoder.inc
        self.down1 = pretrained_encoder.down1
        self.down2 = pretrained_encoder.down2
        self.down3 = pretrained_encoder.down3
        self.down4 = pretrained_encoder.down4
        
        # Keep base model's decoder
        self.up1 = base_model.up1
        self.up2 = base_model.up2
        self.up3 = base_model.up3
        self.up4 = base_model.up4
        self.outc = base_model.outc
        
        # Freeze encoder weights if specified
        if freeze_encoder:
            self._freeze_encoder()
    
    def _freeze_encoder(self):
        """Freeze encoder parameters to prevent updating during training"""
        for param in self.inc.parameters():
            param.requires_grad = False
        
        for param in self.down1.parameters():
            param.requires_grad = False
            
        for param in self.down2.parameters():
            param.requires_grad = False
            
        for param in self.down3.parameters():
            param.requires_grad = False
            
        for param in self.down4.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze encoder for fine-tuning"""
        for param in self.inc.parameters():
            param.requires_grad = True
        
        for param in self.down1.parameters():
            param.requires_grad = True
            
        for param in self.down2.parameters():
            param.requires_grad = True
            
        for param in self.down3.parameters():
            param.requires_grad = True
            
        for param in self.down4.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder path
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        return logits

def load_pretrained_encoder(pretrained_path, model_type='standard', device='cuda'):
    """
    Load pretrained encoder weights
    
    Args:
        pretrained_path: Path to pretrained encoder weights
        model_type: Type of UNet model ('standard' or 'attention')
        device: Device to load weights on
        
    Returns:
        Pretrained encoder model
    """
    # Create base UNet model based on specified type
    if model_type == 'attention' and AttentionUNet is not None:
        try:
            base_model = AttentionUNet(n_channels=3, n_classes=1, bilinear=False)
            print(f"Using AttentionUNet model on device {device}")
        except Exception as e:
            print(f"Error loading AttentionUNet: {e}")
            print("Falling back to standard UNet")
            base_model = UNet(n_channels=3, n_classes=1, bilinear=False)
    else:
        base_model = UNet(n_channels=3, n_classes=1, bilinear=False)
        print(f"Using standard UNet model on device {device}")
    
    # Create encoder from base model
    encoder = UNetEncoder(base_model)
    
    # Check if file exists
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained weights file not found: {pretrained_path}")
    
    # Load pretrained weights
    checkpoint = torch.load(pretrained_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Load from model checkpoint
        encoder.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Direct state dict
        encoder.load_state_dict(checkpoint)
    
    print(f"Loaded pretrained encoder from {pretrained_path}")
    return encoder

def plot_history(history, output_dir, rank=0):
    """Plot training history metrics (only from rank 0)"""
    if rank != 0:
        return
        
    # Create a figure with 4 subplots
    plt.figure(figsize=(18, 12))
    
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(history['train_iou'], label='Train')
    plt.plot(history['val_iou'], label='Validation')
    plt.title('IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    
    # Plot F1 scores for validation and test sets
    plt.subplot(2, 2, 3)
    val_f1_scores = [m['f1'] if m is not None else None for m in history['val_metrics']]
    test_f1_scores = [m['f1'] if m is not None else None for m in history['test_metrics']]
    
    valid_epochs = [i for i, score in enumerate(val_f1_scores) if score is not None]
    valid_val_scores = [score for score in val_f1_scores if score is not None]
    valid_test_scores = [score for score in test_f1_scores if score is not None]
    
    if valid_val_scores:
        plt.plot(valid_epochs, valid_val_scores, marker='o', linestyle='-', color='blue', label='Validation F1')
        plt.plot(valid_epochs, valid_test_scores, marker='s', linestyle='-', color='red', label='Test F1 (Mpox)')
        plt.title('F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
    
    # Plot Precision and Recall
    plt.subplot(2, 2, 4)
    val_precision = [m['precision'] if m is not None else None for m in history['val_metrics']]
    val_recall = [m['recall'] if m is not None else None for m in history['val_metrics']]
    test_precision = [m['precision'] if m is not None else None for m in history['test_metrics']]
    test_recall = [m['recall'] if m is not None else None for m in history['test_metrics']]
    
    valid_val_precision = [score for score in val_precision if score is not None]
    valid_val_recall = [score for score in val_recall if score is not None]
    valid_test_precision = [score for score in test_precision if score is not None]
    valid_test_recall = [score for score in test_recall if score is not None]
    
    if valid_val_precision:
        plt.plot(valid_epochs, valid_val_precision, marker='o', linestyle='-', color='blue', label='Validation Precision')
        plt.plot(valid_epochs, valid_val_recall, marker='o', linestyle='--', color='blue', label='Validation Recall')
        plt.plot(valid_epochs, valid_test_precision, marker='s', linestyle='-', color='red', label='Test Precision')
        plt.plot(valid_epochs, valid_test_recall, marker='s', linestyle='--', color='red', label='Test Recall')
        plt.title('Precision and Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"))
    plt.close()

def save_checkpoint(model, optimizer, epoch, loss, iou, val_metrics, test_metrics, 
                   history, filename, is_best=False, rank=0):
    """Save model checkpoint (only from rank 0)"""
    if rank != 0:
        return
        
    # Unwrap DDP model if needed
    if isinstance(model, DDP):
        model_to_save = model.module
    else:
        model_to_save = model
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': loss,
        'val_iou': iou,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'history': history
    }
    
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")
    
    # If this is the best model, save a copy as best_model.pth
    if is_best:
        best_path = os.path.join(os.path.dirname(filename), "best_model.pth")
        torch.save(checkpoint, best_path)
        print(f"Best model saved to {best_path}")

def train_process(rank, world_size, args):
    """Main training process for distributed training"""
    # Set up distributed training
    setup_distributed(rank, world_size)
    
    # Set device for current process
    device = torch.device(f"cuda:{rank}")
    
    # Create output directory (from rank 0 only)
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(args.output_dir, f"transfer_learning_run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Save arguments
        with open(os.path.join(run_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(args.output_dir, f"transfer_learning_run_{timestamp}")
    
    # Synchronize processes to ensure run_dir is created
    if world_size > 1:
        torch.distributed.barrier()
    
    try:
        # Create datasets
        train_dataset, val_dataset, test_dataset = create_datasets(
            args.ph2_dir, args.mpox_dir, target_size=(args.img_size, args.img_size)
        )
        
        # Create distributed data loaders
        train_loader, train_sampler = create_distributed_dataloader(
            train_dataset, args.batch_size, args.num_workers, world_size, rank, shuffle=True
        )
        
        val_loader, val_sampler = create_distributed_dataloader(
            val_dataset, args.batch_size, args.num_workers, world_size, rank, shuffle=False
        )
        
        test_loader, test_sampler = create_distributed_dataloader(
            test_dataset, args.batch_size, args.num_workers, world_size, rank, shuffle=False
        )
        
        if rank == 0:
            print(f"Training with {len(train_dataset)} images from PH2 dataset")
            print(f"Validating with {len(val_dataset)} images from PH2 dataset")
            print(f"Testing with {len(test_dataset)} images from Mpox dataset")
            print(f"Using {world_size} GPUs for training")
        
        # Load pretrained encoder
        pretrained_encoder = load_pretrained_encoder(
            args.pretrained_weights, args.model_type, device
        )
        
        # Create base UNet model
        if args.model_type == 'attention' and AttentionUNet is not None:
            try:
                base_model = AttentionUNet(n_channels=3, n_classes=1, bilinear=False)
            except Exception as e:
                if rank == 0:
                    print(f"Error loading AttentionUNet: {e}")
                    print("Falling back to standard UNet")
                base_model = UNet(n_channels=3, n_classes=1, bilinear=False)
        else:
            base_model = UNet(n_channels=3, n_classes=1, bilinear=False)
        
        # Create transfer learning model
        model = TransferLearningUNet(
            base_model, pretrained_encoder, freeze_encoder=args.freeze_encoder
        )
        model = model.to(device)
        
        # Wrap model with DDP
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        
        # Create optimizer and loss function
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=args.lr, weight_decay=0.01
        )
        
        # Create loss function
        criterion = EnhancedCombinedLoss(
            bce_weight=0.3, dice_weight=0.5, focal_weight=0.2, reg_weight=0.0005
        )
        
        # Create learning rate scheduler
        if args.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=1e-6
            )
            if rank == 0:
                print("Using CosineAnnealingLR scheduler")
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1, patience=5, verbose=(rank == 0)
            )
            if rank == 0:
                print("Using ReduceLROnPlateau scheduler")
        
        # Initialize training history
        history = {
            'train_loss': [],
            'train_iou': [],
            'val_loss': [],
            'val_iou': [],
            'val_metrics': [],
            'test_metrics': []
        }
        
        # Early stopping variables
        best_val_iou = 0
        best_f1_score = 0
        patience_counter = 0
        
        # Start training
        if rank == 0:
            print("Starting training with transfer learning...")
            
        for epoch in range(args.epochs):
            # Set epoch for samplers
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)
            
            # Unfreeze encoder for fine-tuning after specified epoch
            if epoch == args.fine_tune_epoch:
                if rank == 0:
                    print("Unfreezing encoder for fine-tuning...")
                # Get the module from DDP wrapper
                if hasattr(model.module, 'unfreeze_encoder'):
                    model.module.unfreeze_encoder()
                    
                # Adjust learning rate for fine-tuning
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] / 10
                
                if rank == 0:
                    print(f"Adjusted learning rate to {optimizer.param_groups[0]['lr']}")
            
            # Train for one epoch
            train_loss, train_iou = train_epoch(
                model, train_loader, optimizer, criterion, device, rank, args.accumulation_steps
            )
            
            # Validate
            val_loss, val_iou = validate(model, val_loader, criterion, device, rank)
            
            # Calculate validation metrics every 5 epochs or on final epoch
            if epoch % 5 == 0 or epoch == args.epochs - 1:
                val_metrics = evaluate_clinical_metrics(
                    model, val_loader, threshold=0.7, device=device, 
                    is_distributed=True, rank=rank
                )
                
                # Also evaluate on test set (Mpox)
                test_metrics = evaluate_clinical_metrics(
                    model, test_loader, threshold=0.7, device=device, 
                    is_distributed=True, rank=rank
                )
                
                if rank == 0:
                    print(f"Val Metrics - F1: {val_metrics['f1']:.4f}, "
                         f"Precision: {val_metrics['precision']:.4f}, "
                         f"Recall: {val_metrics['recall']:.4f}")
                    
                    print(f"Test Metrics (Mpox) - F1: {test_metrics['f1']:.4f}, "
                         f"Precision: {test_metrics['precision']:.4f}, "
                         f"Recall: {test_metrics['recall']:.4f}")
            else:
                val_metrics = None
                test_metrics = None
            
            # Update learning rate
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
            
            # Update history (only on rank 0)
            if rank == 0:
                history['train_loss'].append(train_loss)
                history['train_iou'].append(train_iou)
                history['val_loss'].append(val_loss)
                history['val_iou'].append(val_iou)
                history['val_metrics'].append(val_metrics)
                history['test_metrics'].append(test_metrics)
                
                # Print metrics
                print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
            
            # Check if this is the best model based on IoU
            is_best = False
            save_reason = ""
            
            if val_iou > best_val_iou:
                best_val_iou = val_iou
                is_best = True
                patience_counter = 0
                save_reason = "IoU"
            
            # Also check F1 score if metrics were calculated
            if val_metrics is not None and val_metrics['f1'] > best_f1_score:
                best_f1_score = val_metrics['f1']
                is_best = True
                patience_counter = 0
                save_reason = "F1 score"
            else:
                patience_counter += 1
            
            # Save checkpoint if best model or at regular intervals
            if is_best or (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
                checkpoint_path = os.path.join(run_dir, f"model_epoch_{epoch+1}.pth")
                save_checkpoint(
                    model, optimizer, epoch, val_loss, val_iou,
                    val_metrics, test_metrics, history, 
                    checkpoint_path, is_best, rank
                )
                
                if rank == 0 and is_best:
                    print(f"Saved best model checkpoint (Best {save_reason})")
            
            # Early stopping
            if patience_counter >= args.early_stopping_patience:
                if rank == 0:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Final evaluation on Mpox dataset
        if rank == 0:
            print("\nEvaluating final model on Mpox dataset...")
            
        final_test_metrics = evaluate_clinical_metrics(
            model, test_loader, threshold=0.7, device=device, 
            is_distributed=True, rank=rank
        )
        
        if rank == 0:
            print(f"Final Mpox Dataset Metrics:")
            print(f"F1 Score: {final_test_metrics['f1']:.4f}")
            print(f"Precision: {final_test_metrics['precision']:.4f}")
            print(f"Recall: {final_test_metrics['recall']:.4f}")
            
            # Save final metrics
            with open(os.path.join(run_dir, "final_mpox_metrics.json"), 'w') as f:
                json.dump(final_test_metrics, f, indent=4)
            
            # Plot training history
            plot_history(history, run_dir, rank)
            
            # Visualize predictions on Mpox dataset
            print("Generating prediction visualizations on Mpox dataset...")
            vis_dir = os.path.join(run_dir, "mpox_visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            
            # Create a non-distributed loader for visualization to avoid redundant work
            test_dataset_vis = GenericLesionDataset(
                args.mpox_dir, None, transform=None, target_size=(args.img_size, args.img_size),
                aug_transform=A.Compose([
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ]), dataset_type="mpox", use_pseudo_labels=True
            )
            
            test_loader_vis = torch.utils.data.DataLoader(
                test_dataset_vis, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True
            )
            
            visualize_predictions(model, test_loader_vis, device, num_samples=10, output_dir=vis_dir, rank=rank)
            
            print(f"Transfer learning completed. Results saved to {run_dir}")
            
    except Exception as e:
        import traceback
        if rank == 0:
            print(f"Error during training: {e}")
            print(traceback.format_exc())
    finally:
        # Clean up distributed resources
        cleanup_distributed()

def main():
    parser = argparse.ArgumentParser(description='Multi-GPU Transfer Learning with HAM10000 pretrained encoder')
    parser.add_argument('--ph2_dir', required=True, help='Directory containing PH2 dataset')
    parser.add_argument('--mpox_dir', required=True, help='Directory containing Mpox dataset')
    parser.add_argument('--pretrained_weights', required=True, help='Path to pretrained encoder weights')
    parser.add_argument('--output_dir', default='./transfer_learning_outputs', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    parser.add_argument('--fine_tune_epoch', type=int, default=50, help='Epoch to start fine-tuning encoder')
    parser.add_argument('--model_type', default='attention', choices=['attention', 'standard'], 
                       help='UNet model type to use (attention or standard)')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder during initial training')
    parser.add_argument('--scheduler', default='cosine', choices=['cosine', 'plateau'],
                       help='Learning rate scheduler to use')
    parser.add_argument('--world_size', type=int, default=None, 
                       help='Number of GPUs to use (default: all available)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers per GPU for data loading')
    parser.add_argument('--save_interval', type=int, default=10, help='Save checkpoint every N epochs') 
    parser.add_argument('--early_stopping_patience', type=int, default=15, help='Patience for early stopping')
    parser.add_argument('--accumulation_steps', type=int, default=2, 
                       help='Number of batches to accumulate gradients over')
    
    args = parser.parse_args()
    
    # Get world size (number of GPUs)
    if args.world_size is None:
        args.world_size = torch.cuda.device_count()
    
    print(f"Using {args.world_size} GPUs")
    
    # Launch distributed training
    if args.world_size > 1:
        mp.spawn(
            train_process,
            args=(args.world_size, args),
            nprocs=args.world_size,
            join=True
        )
    else:
        # Single GPU training
        train_process(0, 1, args)

if __name__ == "__main__":
    main()
