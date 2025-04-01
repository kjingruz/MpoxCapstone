#!/usr/bin/env python3
"""
Fine-tuning script for MedSAM2 on Mpox lesion segmentation.
This script fine-tunes the pre-trained MedSAM2 model on Mpox lesion data.
"""

import os
import sys
import time
import shutil
import argparse
from pathlib import Path
import json
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Try to import MedSAM2 modules from official repo
try:
    from segment_anything_2 import build_sam2_model
    import yaml  # For reading config files
except ImportError:
    print("Error: segment_anything_2 package not found.")
    print("Make sure you've installed the official MedSAM2 package from bowang-lab.")
    print("Run the setup_medsam2.py script first.")
    sys.exit(1)

def ensure_dir(directory):
    """Make sure directory exists."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory

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

class MedSAM2Dataset(Dataset):
    """Dataset for fine-tuning MedSAM2."""
    def __init__(self, npy_dir, max_dim=256):
        self.npy_dir = Path(npy_dir)
        self.max_dim = max_dim
        
        # Find all image files
        self.image_files = sorted(list(self.npy_dir.glob("*_img.npy")))
        
        # Check if matching mask files exist
        valid_images = []
        for img_file in self.image_files:
            mask_file = img_file.parent / img_file.name.replace("_img.npy", "_mask.npy")
            if mask_file.exists():
                valid_images.append((img_file, mask_file))
        
        self.pairs = valid_images
        print(f"Found {len(self.pairs)} valid image-mask pairs for training.")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        
        # Load image and mask
        img = np.load(img_path)
        mask = np.load(mask_path)
        
        # Ensure mask is binary 
        mask = (mask > 0).astype(np.uint8)
        
        # Convert image to PyTorch tensor
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask).float()
        
        # Handle empty mask case - add a small random region
        if mask_tensor.sum() == 0:
            # Add a small random circle
            h, w = mask.shape
            center_h = random.randint(h//4, 3*h//4)
            center_w = random.randint(w//4, 3*w//4)
            radius = random.randint(5, 15)
            
            # Create a coordinate grid
            y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            distance = torch.sqrt((y - center_h)**2 + (x - center_w)**2)
            mask_tensor = (distance < radius).float()
        
        # Get bounding box from mask
        if mask_tensor.sum() > 0:
            nonzero_indices = torch.nonzero(mask_tensor)
            y_min, x_min = nonzero_indices.min(dim=0)[0]
            y_max, x_max = nonzero_indices.max(dim=0)[0]
            bbox = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float)
        else:
            # Default small box in the center
            h, w = mask.shape
            center_h, center_w = h // 2, w // 2
            size = min(h, w) // 10
            bbox = torch.tensor([center_w - size, center_h - size, 
                                center_w + size, center_h + size], 
                                dtype=torch.float)
        
        # Expand the bbox by 5-10 pixels for better prompting
        h, w = mask.shape
        bbox_shift = random.randint(5, 10)
        bbox[0] = max(0, bbox[0] - bbox_shift)
        bbox[1] = max(0, bbox[1] - bbox_shift)
        bbox[2] = min(w - 1, bbox[2] + bbox_shift)
        bbox[3] = min(h - 1, bbox[3] + bbox_shift)
        
        return {
            'image': img_tensor,
            'mask': mask_tensor.unsqueeze(0),  # Add channel dimension
            'bbox': bbox,
            'filename': img_path.stem.replace("_img", "")
        }

class MedSAM2Loss(nn.Module):
    """Loss function for fine-tuning MedSAM2, similar to the official implementation."""
    def __init__(self, focal_weight=20.0, dice_weight=1.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred_masks, gt_masks):
        # Make sure pred_masks and gt_masks have the same shape
        if pred_masks.shape != gt_masks.shape:
            pred_masks = F.interpolate(
                pred_masks, size=gt_masks.shape[2:], mode="bilinear", align_corners=False
            )
        
        # Calculate focal loss
        focal_loss = self.calculate_focal_loss(pred_masks, gt_masks)
        
        # Calculate dice loss
        dice_loss = self.calculate_dice_loss(pred_masks, gt_masks)
        
        # Combine losses
        total_loss = self.focal_weight * focal_loss + self.dice_weight * dice_loss
        
        return total_loss, focal_loss, dice_loss
    
    def calculate_focal_loss(self, pred_masks, gt_masks, alpha=0.25, gamma=2.0):
        """Calculate focal loss."""
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(pred_masks)
        
        # Prepare targets and weights
        targets = gt_masks.float()
        
        # Calculate focal weights
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weights = (1 - pt) ** gamma
        
        # Apply alpha balancing
        focal_weights = alpha * targets * focal_weights + (1 - alpha) * (1 - targets) * focal_weights
        
        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_masks, targets, reduction="none"
        )
        
        # Apply weights to BCE loss
        focal_loss = focal_weights * bce_loss
        
        # Return mean loss
        return focal_loss.mean()
    
    def calculate_dice_loss(self, pred_masks, gt_masks, smooth=1e-6):
        """Calculate dice loss."""
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(pred_masks)
        
        # Calculate dice coefficient
        numerator = 2 * (probs * gt_masks).sum(dim=(1, 2, 3)) + smooth
        denominator = probs.sum(dim=(1, 2, 3)) + gt_masks.sum(dim=(1, 2, 3)) + smooth
        dice = numerator / denominator
        
        # Return 1 - dice as loss
        return 1 - dice.mean()

def evaluate_model(model, dataloader, device, loss_fn=None):
    """Evaluate model on validation data."""
    model.eval()
    total_loss = 0
    total_focal_loss = 0
    total_dice_loss = 0
    total_iou = 0
    count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            boxes = batch['bbox'].to(device)
            
            # Forward pass
            pred_masks, _ = model.forward_custom(images, boxes)
            
            # Calculate loss if loss function provided
            if loss_fn:
                loss, focal_loss, dice_loss = loss_fn(pred_masks, masks)
                total_loss += loss.item()
                total_focal_loss += focal_loss.item()
                total_dice_loss += dice_loss.item()
            
            # Calculate IoU
            pred_binary = (torch.sigmoid(pred_masks) > 0.5).float()
            intersection = (pred_binary * masks).sum(dim=(1, 2, 3))
            union = pred_binary.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) - intersection
            batch_iou = (intersection / (union + 1e-6)).mean().item()
            total_iou += batch_iou
            
            count += 1
    
    # Calculate averages
    avg_loss = total_loss / count if count > 0 else float('inf')
    avg_focal_loss = total_focal_loss / count if count > 0 else float('inf')
    avg_dice_loss = total_dice_loss / count if count > 0 else float('inf')
    avg_iou = total_iou / count if count > 0 else 0
    
    return avg_loss, avg_focal_loss, avg_dice_loss, avg_iou

def visualize_predictions(model, dataloader, device, output_dir, num_samples=4):
    """Visualize model predictions for qualitative evaluation."""
    model.eval()
    vis_dir = ensure_dir(os.path.join(output_dir, "visualizations"))
    
    # Get a few random samples
    samples = []
    for batch in dataloader:
        for i in range(min(len(batch['image']), num_samples - len(samples))):
            samples.append({
                'image': batch['image'][i:i+1].to(device),
                'mask': batch['mask'][i:i+1].to(device),
                'bbox': batch['bbox'][i:i+1].to(device),
                'filename': batch['filename'][i]
            })
            if len(samples) >= num_samples:
                break
        if len(samples) >= num_samples:
            break
    
    # Generate predictions
    for i, sample in enumerate(samples):
        with torch.no_grad():
            # Forward pass
            pred_masks, _ = model.forward_custom(sample['image'], sample['bbox'])
            
            # Convert to numpy for visualization
            image = sample['image'].cpu().squeeze().permute(1, 2, 0).numpy()
            mask_gt = sample['mask'].cpu().squeeze().numpy()
            mask_pred = torch.sigmoid(pred_masks).cpu().squeeze().numpy() > 0.5
            
            # Create figure
            plt.figure(figsize=(15, 5))
            
            # Plot original image
            plt.subplot(1, 3, 1)
            plt.imshow(image)
            plt.title("Original Image")
            plt.axis('off')
            
            # Plot ground truth mask
            plt.subplot(1, 3, 2)
            plt.imshow(mask_gt, cmap='gray')
            plt.title("Ground Truth")
            plt.axis('off')
            
            # Plot predicted mask
            plt.subplot(1, 3, 3)
            plt.imshow(mask_pred, cmap='gray')
            plt.title("Prediction")
            plt.axis('off')
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"sample_{i+1}_{sample['filename']}.png"))
            plt.close()
    
    print(f"Saved {len(samples)} visualization samples to {vis_dir}")

def finetune_medsam2(npy_dir, output_dir, sam2_checkpoint, model_cfg=None, 
                    batch_size=8, num_epochs=50, learning_rate=1e-5, 
                    device="cuda", seed=42, val_split=0.1, patience=10):
    """Fine-tune MedSAM2 on Mpox lesion data."""
    # Set random seed for reproducibility
    set_seed(seed)
    
    # Create output directory
    run_dir = ensure_dir(os.path.join(output_dir, f"medsam2_finetune_{int(time.time())}"))
    
    # Set up tensorboard writer if available
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=os.path.join(run_dir, "tensorboard"))
        use_tensorboard = True
    except ImportError:
        use_tensorboard = False
        print("TensorBoard not available. Install it with: pip install tensorboard")
    
    # Load model config
    if model_cfg and model_cfg.endswith(".yaml"):
        with open(model_cfg, 'r') as f:
            cfg = yaml.safe_load(f)
            backbone_name = cfg.get('backbone_name', 'hiera_base_plus')
    else:
        # Guess model type from checkpoint name
        if "tiny" in sam2_checkpoint:
            backbone_name = "hiera_tiny"
        elif "small" in sam2_checkpoint:
            backbone_name = "hiera_small"
        elif "base_plus" in sam2_checkpoint:
            backbone_name = "hiera_base_plus"
        elif "large" in sam2_checkpoint:
            backbone_name = "hiera_large"
        else:
            print("Could not determine model type from checkpoint name.")
            print("Using default: hiera_base_plus")
            backbone_name = "hiera_base_plus"
    
    # Build the SAM2 model
    print(f"Building SAM2 model with {backbone_name} backbone")
    sam2_model = build_sam2_model(
        checkpoint=sam2_checkpoint,
        backbone_name=backbone_name
    )
    
    # Add a custom forward method for training with boxes
    def forward_custom(self, images, boxes):
        """Custom forward pass taking batch of images and boxes directly."""
        # Get image embeddings
        image_embeddings = self.image_encoder(images)
        
        # Prepare prompt encoder inputs
        box_torch = boxes
        
        # Convert boxes to masks using prompt encoder
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        
        # Predict masks
        mask_predictions, _ = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        
        return mask_predictions, image_embeddings
    
    # Add custom forward method to the model
    sam2_model.forward_custom = forward_custom.__get__(sam2_model, sam2_model.__class__)
    
    # Move model to device
    sam2_model.to(device)
    
    # Create dataset
    dataset = MedSAM2Dataset(npy_dir)
    
    # Split into train and validation
    dataset_size = len(dataset)
    val_size = max(1, int(dataset_size * val_split))
    train_size = dataset_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"Dataset split: {train_size} training, {val_size} validation")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    # Create loss function
    loss_fn = MedSAM2Loss(focal_weight=20.0, dice_weight=1.0)
    
    # Create optimizer
    # Only train the decoder and prompt encoder, keep image encoder frozen
    params_to_train = []
    total_params = 0
    trainable_params = 0
    
    # First, freeze all parameters
    for param in sam2_model.parameters():
        param.requires_grad = False
        total_params += param.numel()
    
    # Unfreeze decoder parameters
    for param in sam2_model.mask_decoder.parameters():
        param.requires_grad = True
        params_to_train.append(param)
        trainable_params += param.numel()
    
    # Unfreeze prompt encoder parameters
    for param in sam2_model.prompt_encoder.parameters():
        param.requires_grad = True
        params_to_train.append(param)
        trainable_params += param.numel()
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params / total_params:.2%})")
    
    # Create optimizer
    optimizer = optim.AdamW(params_to_train, lr=learning_rate, weight_decay=0.01)
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_focal_loss': [],
        'train_dice_loss': [],
        'train_iou': [],
        'val_loss': [],
        'val_focal_loss': [],
        'val_dice_loss': [],
        'val_iou': [],
        'lr': []
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_val_iou = 0
    best_epoch = 0
    patience_counter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        sam2_model.train()
        epoch_loss = 0
        epoch_focal_loss = 0
        epoch_dice_loss = 0
        epoch_iou = 0
        batch_count = 0
        
        # Use tqdm for progress bar
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch in pbar:
                # Get data
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                boxes = batch['bbox'].to(device)
                
                # Forward pass
                pred_masks, _ = sam2_model.forward_custom(images, boxes)
                
                # Calculate loss
                loss, focal_loss, dice_loss = loss_fn(pred_masks, masks)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                epoch_focal_loss += focal_loss.item()
                epoch_dice_loss += dice_loss.item()
                
                # Calculate IoU
                pred_binary = (torch.sigmoid(pred_masks) > 0.5).float()
                intersection = (pred_binary * masks).sum(dim=(1, 2, 3))
                union = pred_binary.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) - intersection
                batch_iou = (intersection / (union + 1e-6)).mean().item()
                epoch_iou += batch_iou
                
                batch_count += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': loss.item(),
                    'iou': batch_iou
                })
        
        # Calculate average metrics
        train_loss = epoch_loss / batch_count
        train_focal_loss = epoch_focal_loss / batch_count
        train_dice_loss = epoch_dice_loss / batch_count
        train_iou = epoch_iou / batch_count
        
        # Validation phase
        val_loss, val_focal_loss, val_dice_loss, val_iou = evaluate_model(
            sam2_model, val_loader, device, loss_fn
        )
        
        # Update scheduler
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_focal_loss'].append(train_focal_loss)
        history['train_dice_loss'].append(train_dice_loss)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_focal_loss'].append(val_focal_loss)
        history['val_dice_loss'].append(val_dice_loss)
        history['val_iou'].append(val_iou)
        history['lr'].append(current_lr)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Focal: {train_focal_loss:.4f}, Dice: {train_dice_loss:.4f}, IoU: {train_iou:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Focal: {val_focal_loss:.4f}, Dice: {val_dice_loss:.4f}, IoU: {val_iou:.4f}")
        print(f"  Learning Rate: {current_lr}")
        
        # Write to tensorboard if available
        if use_tensorboard:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/train_focal', train_focal_loss, epoch)
            writer.add_scalar('Loss/train_dice', train_dice_loss, epoch)
            writer.add_scalar('IoU/train', train_iou, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Loss/val_focal', val_focal_loss, epoch)
            writer.add_scalar('Loss/val_dice', val_dice_loss, epoch)
            writer.add_scalar('IoU/val', val_iou, epoch)
            writer.add_scalar('LR', current_lr, epoch)
        
        # Check if this is the best model
        is_best = False
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_epoch = epoch
            patience_counter = 0
            is_best = True
        else:
            patience_counter += 1
        
        # Save checkpoint
        checkpoint_path = os.path.join(run_dir, "medsam2_model_latest.pth")
        torch.save(sam2_model.state_dict(), checkpoint_path)
        
        # Save best model
        if is_best:
            best_model_path = os.path.join(run_dir, "medsam2_model_best.pth")
            shutil.copy(checkpoint_path, best_model_path)
            print(f"  Saved best model with IoU: {best_val_iou:.4f}")
        
        # Check early stopping
        if patience_counter >= patience:
            print(f"Early stopping after {epoch+1} epochs without improvement.")
            break
        
        # Generate visualizations every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == num_epochs - 1:
            visualize_predictions(
                sam2_model, val_loader, device, 
                os.path.join(run_dir, f"epoch_{epoch+1}")
            )
    
    # Load best model
    best_model_path = os.path.join(run_dir, "medsam2_model_best.pth")
    sam2_model.load_state_dict(torch.load(best_model_path))
    
    # Final evaluation
    final_val_loss, final_val_focal_loss, final_val_dice_loss, final_val_iou = evaluate_model(
        sam2_model, val_loader, device, loss_fn
    )
    
    print("\nFinal Model Performance:")
    print(f"  Validation Loss: {final_val_loss:.4f}")
    print(f"  Validation IoU: {final_val_iou:.4f}")
    print(f"  Best Epoch: {best_epoch + 1}")
    
    # Generate final visualizations
    visualize_predictions(
        sam2_model, val_loader, device, 
        os.path.join(run_dir, "final")
    )
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot IoU
    plt.subplot(1, 3, 2)
    plt.plot(history['train_iou'], label='Train')
    plt.plot(history['val_iou'], label='Validation')
    plt.title('IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    
    # Plot learning rate
    plt.subplot(1, 3, 3)
    plt.plot(history['lr'])
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "training_history.png"))
    plt.close()
    
    # Save training history
    with open(os.path.join(run_dir, "training_history.json"), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save final model in a standard location
    final_model_path = os.path.join(output_dir, "medsam2_mpox_finetuned.pth")
    shutil.copy(best_model_path, final_model_path)
    
    print(f"Fine-tuning completed. Results saved to {run_dir}")
    print(f"Final model saved to {final_model_path}")
    
    return sam2_model, history, final_model_path

def main():
    parser = argparse.ArgumentParser(description="Fine-tune MedSAM2 on Mpox lesion data")
    parser.add_argument("--npy_dir", required=True, 
                        help="Directory containing npy files prepared for training")
    parser.add_argument("--output_dir", default="./medsam2_finetune", 
                        help="Output directory for fine-tuning results")
    parser.add_argument("--sam2_checkpoint", required=True, 
                        help="Path to SAM2 checkpoint")
    parser.add_argument("--model_cfg", default=None, 
                        help="Path to model config YAML file")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=50, 
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, 
                        help="Initial learning rate")
    parser.add_argument("--val_split", type=float, default=0.1, 
                        help="Validation set fraction")
    parser.add_argument("--patience", type=int, default=10, 
                        help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], 
                        help="Device to run training on")
    parser.add_argument("--resume", default=None, 
                        help="Path to checkpoint to resume training from")
    
    args = parser.parse_args()
    
    # Check if CUDA is available if requested
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Using CPU instead.")
        args.device = "cpu"
        
    # Check if npy directory exists and contains data
    npy_dir = Path(args.npy_dir)
    if not npy_dir.exists():
        print(f"Error: Directory {npy_dir} does not exist.")
        sys.exit(1)
    
    npy_files = list(npy_dir.glob("*_img.npy"))
    if len(npy_files) == 0:
        print(f"Error: No npy files found in {npy_dir}")
        print("Run mpox_data_prep.py with --mode training first.")
        sys.exit(1)
    
    # Fine-tune the model
    model, history, final_model_path = finetune_medsam2(
        args.npy_dir,
        args.output_dir,
        args.sam2_checkpoint,
        args.model_cfg,
        args.batch_size,
        args.num_epochs,
        args.learning_rate,
        args.device,
        args.seed,
        args.val_split,
        args.patience
    )
    
    # Save configuration for reference
    config = vars(args)
    config['timestamp'] = str(Path.ctime(Path.cwd()))
    config['final_model_path'] = final_model_path
    
    with open(os.path.join(args.output_dir, "finetune_config.json"), 'w') as f:
        json.dump(config, f, indent=2)

if __name__ == "__main__":
    main()
