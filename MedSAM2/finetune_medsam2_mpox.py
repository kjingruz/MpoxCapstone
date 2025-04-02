#!/usr/bin/env python3
"""
Fine-tune SAM2 model on Mpox lesion data.
This script fine-tunes the SAM2 base_plus model for Mpox lesion segmentation.
"""

import os
import sys
import numpy as np
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import time
import shutil
import json
from datetime import datetime
import cv2
from sam2.build_sam import build_sam2
from sam2.utils.transforms import SAM2Transforms

def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def ensure_dir(directory):
    """Make sure directory exists."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0, 0, 0, 0), lw=2)
    )

class MpoxDataset(Dataset):
    """Dataset for fine-tuning SAM2 on Mpox data."""
    def __init__(self, data_root, bbox_shift=20, transform=None):
        self.data_root = data_root
        self.gt_path = os.path.join(data_root, "gts")
        self.img_path = os.path.join(data_root, "imgs")
        self.gt_files = sorted(list(Path(self.gt_path).glob("*.npy")))
        self.img_files = sorted(list(Path(self.img_path).glob("*.npy")))
        self.bbox_shift = bbox_shift
        
        # Match image and ground truth files
        self.paired_files = []
        for gt_file in self.gt_files:
            img_file = os.path.join(self.img_path, gt_file.name)
            if os.path.exists(img_file):
                self.paired_files.append((Path(img_file), gt_file))
        
        self.transform = transform if transform else SAM2Transforms(resolution=1024, mask_threshold=0)
        print(f"Found {len(self.paired_files)} paired image-mask files.")
    
    def __len__(self):
        return len(self.paired_files)
    
    def __getitem__(self, idx):
        img_path, gt_path = self.paired_files[idx]
        
        # Load image and mask
        img = np.load(str(img_path), allow_pickle=True)
        gt = np.load(str(gt_path), allow_pickle=True)
        
        # Ensure binary mask
        gt = (gt > 0).astype(np.uint8)
        
        # Get bounding box from mask with random shift
        if gt.sum() > 0:
            y_indices, x_indices = np.where(gt > 0)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            
            # Add random shift to bounding box
            h, w = gt.shape
            x_min = max(0, x_min - random.randint(0, self.bbox_shift))
            x_max = min(w-1, x_max + random.randint(0, self.bbox_shift))
            y_min = max(0, y_min - random.randint(0, self.bbox_shift))
            y_max = min(h-1, y_max + random.randint(0, self.bbox_shift))
            
            bbox = np.array([x_min, y_min, x_max, y_max])
        else:
            # Create default box for empty masks
            h, w = gt.shape
            center_x, center_y = w // 2, h // 2
            size = min(h, w) // 10
            bbox = np.array([center_x - size, center_y - size, center_x + size, center_y + size])
        
        # Apply SAM2 transformations to image
        img_1024 = self.transform(img.copy())
        # Scale bbox to 1024x1024 (from 256x256)
        bbox_1024 = bbox * (1024 / 256)
        
        return (
            img_1024,  # [3, 1024, 1024]
            torch.tensor(gt[None, :, :]).float(),  # [1, 256, 256]
            torch.tensor(bbox_1024).float(),
            img_path.stem
        )

class MedSAM2(nn.Module):
    """MedSAM2 model for fine-tuning on Mpox data."""
    def __init__(self, model):
        super().__init__()
        self.sam2_model = model
        
        # Freeze prompt encoder
        for param in self.sam2_model.sam_prompt_encoder.parameters():
            param.requires_grad = False
    
    def forward(self, image, box):
        """
        Forward pass for training.
        
        Args:
            image: Tensor of shape (B, 3, 1024, 1024)
            box: Tensor or Array of shape (B, 4) with coordinates [x_min, y_min, x_max, y_max]
        
        Returns:
            Mask logits of shape (B, 1, 256, 256)
        """
        _features = self._image_encoder(image)
        img_embed, high_res_features = _features["image_embed"], _features["high_res_feats"]
        
        # Process boxes into SAM2 format (no gradients for prompt encoder)
        with torch.no_grad():
            # Ensure box is a tensor with the right device
            if not isinstance(box, torch.Tensor):
                box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            else:
                box_torch = box.to(device=image.device)
                
            if len(box_torch.shape) == 2:
                box_coords = box_torch.reshape(-1, 2, 2)  # (B, 4) to (B, 2, 2)
                box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=image.device)
                box_labels = box_labels.repeat(box_torch.size(0), 1)
            concat_points = (box_coords, box_labels)
            
            sparse_embeddings, dense_embeddings = self.sam2_model.sam_prompt_encoder(
                points=concat_points,
                boxes=None,
                masks=None,
            )
        
        # Get mask predictions
        low_res_masks_logits, iou_predictions, sam_tokens_out, object_score_logits = self.sam2_model.sam_mask_decoder(
            image_embeddings=img_embed,
            image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        
        return low_res_masks_logits
    
    def _image_encoder(self, input_image):
        """Extract image features using the SAM2 image encoder."""
        backbone_out = self.sam2_model.forward_image(input_image)
        _, vision_feats, _, _ = self.sam2_model._prepare_backbone_features(backbone_out)
        
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.sam2_model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.sam2_model.no_mem_embed
        
        bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
        feats = [
            feat.permute(1, 2, 0).view(input_image.size(0), -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1])
        ][::-1]
        
        _features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        return _features

def dice_loss(pred, target, smooth=1e-5):
    """Calculate Dice loss."""
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def evaluate_model(model, dataloader, device):
    """Evaluate model on validation data."""
    model.eval()
    total_dice = 0
    total_iou = 0
    count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images, masks, boxes, _ = batch
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass
            mask_preds = model(images, boxes.numpy())
            
            # Convert predictions to binary masks
            pred_masks = (torch.sigmoid(mask_preds) > 0.5).float()
            
            # Calculate Dice score
            intersection = (pred_masks * masks).sum(dim=(1, 2, 3))
            union = pred_masks.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
            dice_score = (2. * intersection) / (union + 1e-5)
            total_dice += dice_score.mean().item()
            
            # Calculate IoU
            intersection = (pred_masks * masks).sum(dim=(1, 2, 3))
            union = pred_masks.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) - intersection
            iou_score = intersection / (union + 1e-5)
            total_iou += iou_score.mean().item()
            
            count += 1
    
    return total_dice / count, total_iou / count

def visualize_predictions(model, dataloader, device, output_dir, num_samples=4):
    """Generate visualization of model predictions."""
    model.eval()
    vis_dir = ensure_dir(os.path.join(output_dir, "visualizations"))
    
    # Get transform for visualization
    inv_transform = dataloader.dataset.transform.inverse
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
            
            images, masks, boxes, names = batch
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass
            mask_preds = model(images, boxes.numpy())
            pred_masks = (torch.sigmoid(mask_preds) > 0.5).float()
            
            # Convert tensors to numpy
            for j in range(images.size(0)):
                if i*images.size(0) + j >= num_samples:
                    break
                
                # Get data for visualization
                image = inv_transform(images[j].cpu()).permute(1, 2, 0).numpy()
                mask_gt = masks[j].cpu().squeeze().numpy()
                mask_pred = pred_masks[j].cpu().squeeze().numpy()
                box = boxes[j].numpy()
                name = names[j]
                
                # Create figure
                fig, axs = plt.subplots(1, 3, figsize=(18, 6))
                
                # Plot original image with box
                axs[0].imshow(image)
                show_box(box, axs[0])
                axs[0].set_title(f"Image: {name}")
                axs[0].axis('off')
                
                # Plot ground truth mask
                axs[1].imshow(image)
                show_mask(mask_gt, axs[1])
                axs[1].set_title("Ground Truth")
                axs[1].axis('off')
                
                # Plot predicted mask
                axs[2].imshow(image)
                show_mask(mask_pred, axs[2])
                axs[2].set_title("Prediction")
                axs[2].axis('off')
                
                # Save figure
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, f"sample_{i*images.size(0)+j+1}.png"))
                plt.close()
    
    print(f"Saved {num_samples} visualization samples to {vis_dir}")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune SAM2 on Mpox lesion data")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing npy files with gts and imgs subfolders")
    parser.add_argument("--output_dir", type=str, default="./medsam2_mpox_finetune",
                       help="Output directory for fine-tuned model")
    parser.add_argument("--sam2_checkpoint", type=str, required=True,
                       help="Path to SAM2 checkpoint file")
    parser.add_argument("--model_cfg", type=str, default="sam2_hiera_b+.yaml",
                       help="Path or name of model config file")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for training")
    parser.add_argument("--val_split", type=float, default=0.2,
                       help="Validation split ratio")
    parser.add_argument("--num_epochs", type=int, default=30,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--bbox_shift", type=int, default=10,
                       help="Random shift for bounding boxes")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loader workers")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda or cpu)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--vis_samples", type=int, default=8,
                       help="Number of samples to visualize")
    
    args = parser.parse_args()
    
    # Set up directories and seed
    set_seed(args.seed)
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    model_save_path = os.path.join(args.output_dir, f"mpox_medsam2_{run_id}")
    ensure_dir(model_save_path)
    
    # Save the script for reproducibility
    shutil.copyfile(__file__, os.path.join(model_save_path, f"{run_id}_{os.path.basename(__file__)}"))
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Load model
    print(f"Building SAM2 model from {args.sam2_checkpoint}")
    sam2_model = build_sam2(args.model_cfg, args.sam2_checkpoint, device=device)
    medsam2_model = MedSAM2(model=sam2_model)
    import torch.nn as nn
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        original_model = medsam2_model
        medsam2_model = nn.DataParallel(medsam2_model)
    else:
        original_model = medsam2_model
    medsam2_model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in medsam2_model.parameters())
    trainable_params = sum(p.numel() for p in medsam2_model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
    
    # Create dataset
    dataset = MpoxDataset(args.data_dir, bbox_shift=args.bbox_shift)
    
    # Split into train and validation
    dataset_size = len(dataset)
    val_size = int(dataset_size * args.val_split)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"Dataset split: {train_size} training, {val_size} validation")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Visualize a few samples for verification
    num_samples = 4
    samples = []
    for batch in train_loader:
        for i in range(min(len(batch[0]), num_samples - len(samples))):
            samples.append({
                'image': batch[0][i:i+1],
                'mask': batch[1][i:i+1],
                'bbox': batch[2][i:i+1],
                'name': batch[3][i]
            })
            if len(samples) >= num_samples:
                break
        if len(samples) >= num_samples:
            break
    
    # Create visualization of training samples
    vis_dir = ensure_dir(os.path.join(model_save_path, "data_samples"))
    
    # Skip the inverse transform - just normalize the image for display
    def simple_normalize(img_tensor):
        # Simple normalization for visualization
        img = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # Normalize to [0,1]
        return img
    
    for i, sample in enumerate(samples):
        image = simple_normalize(sample['image'])
        mask = sample['mask'].squeeze().numpy()
        bbox = sample['bbox'].squeeze().numpy()
        name = sample['name']
        
        plt.figure(figsize=(12, 4))
        
        # Original image with box
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        show_box(bbox, plt.gca())
        plt.title(f"Image: {name}")
        plt.axis('off')
        
        # Mask
        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='gray')
        plt.title("Mask")
        plt.axis('off')
        
        # Overlay
        plt.subplot(1, 3, 3)
        plt.imshow(image)
        show_mask(mask, plt.gca())
        plt.title("Overlay")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"sample_{i+1}.png"))
        plt.close()
    
    # Setup optimizer and losses
    param_groups = [
        {'params': original_model.sam2_model.image_encoder.parameters()},
        {'params': original_model.sam2_model.sam_mask_decoder.parameters()}
    ]
    
    optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate, weight_decay=args.weight_decay)
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    
    # Setup learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_dice = 0
    train_losses = []
    val_dices = []
    val_ious = []
    
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        medsam2_model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_dice = checkpoint.get('best_val_dice', 0)
        train_losses = checkpoint.get('train_losses', [])
        val_dices = checkpoint.get('val_dices', [])
        val_ious = checkpoint.get('val_ious', [])
        print(f"Resuming from epoch {start_epoch}, best val dice: {best_val_dice:.4f}")
    
    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        # Training phase
        medsam2_model.train()
        epoch_loss = 0
        batch_count = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}") as pbar:
            for batch in pbar:
                images, masks, boxes, _ = batch
                images, masks = images.to(device), masks.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                mask_preds = medsam2_model(images, boxes)
                
                # Calculate loss (Dice + BCE)
                loss = dice_loss(mask_preds, masks) + ce_loss(mask_preds, masks)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                batch_count += 1
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / batch_count
        train_losses.append(avg_epoch_loss)
        
        # Validation phase
        val_dice, val_iou = evaluate_model(medsam2_model, val_loader, device)
        val_dices.append(val_dice)
        val_ious.append(val_iou)
        
        # Update learning rate
        scheduler.step(1 - val_dice)  # Use 1-dice as the metric to minimize
        
        # Print results
        print(f"Epoch {epoch+1}/{args.num_epochs}:")
        print(f"  Train Loss: {avg_epoch_loss:.4f}")
        print(f"  Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save checkpoint
        is_best = val_dice > best_val_dice
        if is_best:
            best_val_dice = val_dice
        
        checkpoint = {
            'epoch': epoch,
            'model': medsam2_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_dice': best_val_dice,
            'train_losses': train_losses,
            'val_dices': val_dices,
            'val_ious': val_ious,
            'args': vars(args)
        }
        
        # Save latest model
        torch.save(checkpoint, os.path.join(model_save_path, "medsam2_model_latest.pth"))
        
        # Save best model
        if is_best:
            torch.save(checkpoint, os.path.join(model_save_path, "medsam2_model_best.pth"))
            print(f"  New best model saved! (Dice: {val_dice:.4f})")
            
            # Generate visualizations for best model
            visualize_predictions(
                medsam2_model, val_loader, device, 
                os.path.join(model_save_path, f"best_epoch_{epoch+1}"),
                args.vis_samples
            )
        
        # Plot training progress
        plt.figure(figsize=(15, 5))
        
        # Plot loss
        plt.subplot(1, 3, 1)
        plt.plot(train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        # Plot Dice
        plt.subplot(1, 3, 2)
        plt.plot(val_dices)
        plt.title('Validation Dice')
        plt.xlabel('Epoch')
        plt.ylabel('Dice')
        
        # Plot IoU
        plt.subplot(1, 3, 3)
        plt.plot(val_ious)
        plt.title('Validation IoU')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        
        plt.tight_layout()
        plt.savefig(os.path.join(model_save_path, "training_progress.png"))
        plt.close()
    
    # Save final model as standalone file for inference
    torch.save(
        medsam2_model.state_dict(), 
        os.path.join(args.output_dir, "medsam2_mpox_final.pth")
    )
    
    print(f"Training completed. Final models saved to {args.output_dir}")
    
    # Generate final visualizations
    print("Generating final visualizations...")
    visualize_predictions(
        medsam2_model, val_loader, device, 
        os.path.join(model_save_path, "final"),
        args.vis_samples
    )

if __name__ == "__main__":
    main()
