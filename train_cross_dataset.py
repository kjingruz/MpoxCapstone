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
import cv2
import json

# Import our model and dataset
from unet_model import UNet
from attention_unet import AttentionUNet
from cross_dataset_loader import create_cross_dataset_loaders
from self_supervised import pretrain_with_self_supervision

# Dice loss for better segmentation performance
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

def evaluate_clinical_metrics(model, dataloader, threshold=0.7, device='cuda'):
    """Calculate simple metrics without connected components"""
    import numpy as np
    
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
    
    # Calculate metrics
    precision = total_tp_pixels / (total_tp_pixels + total_fp_pixels) if (total_tp_pixels + total_fp_pixels) > 0 else 0
    recall = total_tp_pixels / (total_tp_pixels + total_fn_pixels) if (total_tp_pixels + total_fn_pixels) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Estimate average lesions based on reasonable assumptions
    # Assuming average lesion is about 500 pixels
    est_lesion_size = 500
    est_true_lesions = total_tp_pixels / est_lesion_size if est_lesion_size > 0 else 0
    est_pred_lesions = (total_tp_pixels + total_fp_pixels) / est_lesion_size if est_lesion_size > 0 else 0
    
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
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    epoch_iou = 0
    
    # Wrap dataloader with tqdm for progress bar
    with tqdm(dataloader, unit="batch") as tepoch:
        for batch in tepoch:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Ensure masks have channel dimension
            if len(masks.shape) == 3:
                masks = masks.unsqueeze(1)
            
            # Ensure float type
            masks = masks.float()
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, masks, model)
            
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

# Main training loop with cross-dataset evaluation
def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler,
                num_epochs, device, checkpoint_dir, log_interval=1,
                early_stopping_patience=15):
    # Initialize training history
    history = {
        'train_loss': [],
        'train_iou': [],
        'val_loss': [],
        'val_iou': [],
        'val_metrics': [],
        'test_metrics': []  # Track test metrics over time
    }
    
    # Early stopping variables
    best_val_iou = 0
    best_f1_score = 0
    patience_counter = 0
    
    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Start training
    print("Starting training with cross-dataset evaluation...")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train for one epoch
        train_loss, train_iou = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_iou = validate(model, val_loader, criterion, device)
        
        # Calculate validation metrics every 5 epochs
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            val_metrics = evaluate_clinical_metrics(model, val_loader, threshold=0.7, device=device)
            history['val_metrics'].append(val_metrics)
            
            # Also evaluate on test set (cross-dataset)
            test_metrics = evaluate_clinical_metrics(model, test_loader, threshold=0.7, device=device)
            history['test_metrics'].append(test_metrics)
            
            print(f"Val Metrics - F1: {val_metrics['f1']:.4f}, "
                  f"Precision: {val_metrics['precision']:.4f}, "
                  f"Recall: {val_metrics['recall']:.4f}")
            
            print(f"Test Metrics (Cross-Dataset) - F1: {test_metrics['f1']:.4f}, "
                  f"Precision: {test_metrics['precision']:.4f}, "
                  f"Recall: {test_metrics['recall']:.4f}")
        else:
            history['val_metrics'].append(None)
            history['test_metrics'].append(None)
        
        # Update learning rate
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
        
        # Check if this is the best model based on IoU
        should_save = False
        
        # Check if this is the best model based on IoU
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            should_save = True
            patience_counter = 0
            save_reason = "IoU"
        
        # Also check F1 score if metrics were calculated
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            if val_metrics['f1'] > best_f1_score:
                best_f1_score = val_metrics['f1']
                should_save = True
                patience_counter = 0
                save_reason = "F1 score"
        
        # Save checkpoint if this is the best model
        if should_save:
            # Save checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f"best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_iou,
                'val_metrics': val_metrics if epoch % 5 == 0 or epoch == num_epochs - 1 else None,
                'test_metrics': test_metrics if epoch % 5 == 0 or epoch == num_epochs - 1 else None,
                'history': history
            }, checkpoint_path)
            
            print(f"Saved best model checkpoint (Best {save_reason})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Save intermediate checkpoint every log_interval epochs
        if (epoch + 1) % log_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_iou,
                'val_metrics': val_metrics if epoch % 5 == 0 or epoch == num_epochs - 1 else None,
                'test_metrics': test_metrics if epoch % 5 == 0 or epoch == num_epochs - 1 else None,
                'history': history
            }, checkpoint_path)
    
    # Plot training history
    plot_training_history(history, checkpoint_dir)
    
    return history

def plot_training_history(history, output_dir):
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
        plt.plot(valid_epochs, valid_test_scores, marker='s', linestyle='-', color='red', label='Test F1 (Cross-Dataset)')
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
    
    # Also save as separate plots for better detail
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_history.png"))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_iou'], label='Train')
    plt.plot(history['val_iou'], label='Validation')
    plt.title('IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "iou_history.png"))
    plt.close()
    
    if valid_val_scores:
        plt.figure(figsize=(10, 6))
        plt.plot(valid_epochs, valid_val_scores, marker='o', linestyle='-', color='blue', label='Validation F1')
        plt.plot(valid_epochs, valid_test_scores, marker='s', linestyle='-', color='red', label='Test F1 (Cross-Dataset)')
        plt.title('F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.savefig(os.path.join(output_dir, "f1_history.png"))
        plt.close()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train U-Net with Cross-Dataset Evaluation')
    parser.add_argument('--ph2_dir', required=True, help='Directory containing PH2 dataset')
    parser.add_argument('--mpox_dir', required=True, help='Directory containing Mpox dataset')
    parser.add_argument('--output_dir', default='./outputs', help='Output directory for models and results')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=256, help='Image size (square)')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--model_type', default='attention', choices=['attention', 'standard'], 
                       help='UNet model type to use (attention or standard)')
    parser.add_argument('--scheduler', default='cosine', choices=['cosine', 'plateau'],
                       help='Learning rate scheduler to use')
    parser.add_argument('--threshold', type=float, default=0.7, 
                       help='Confidence threshold for predictions')
    parser.add_argument('--use_pretrained', action='store_true', help='Use self-supervised pretraining')
    parser.add_argument('--pretrain_method', default='simsiam', choices=['rotation', 'contrastive', 'simsiam'], 
                       help='Self-supervised pretraining method')
    parser.add_argument('--pretrain_epochs', type=int, default=50, help='Number of pretraining epochs')
    
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
    checkpoint_dir = os.path.join(args.output_dir, f"cross_dataset_run_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Log arguments
    with open(os.path.join(checkpoint_dir, "args.txt"), "w") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    # Create data loaders for cross-dataset training
    train_loader, val_loader, test_loader = create_cross_dataset_loaders(
        args.ph2_dir, args.mpox_dir, batch_size=args.batch_size, 
        target_size=(args.img_size, args.img_size), num_workers=args.num_workers
    )
    
    print(f"Training with {len(train_loader.dataset)} images from PH2 dataset")
    print(f"Validating with {len(val_loader.dataset)} images from PH2 dataset")
    print(f"Testing with {len(test_loader.dataset)} images from Mpox dataset")
    
    # Create model based on argument
    if args.model_type == 'attention':
        try:
            model = AttentionUNet(n_channels=3, n_classes=1, bilinear=False)
            print("Using AttentionUNet model")
        except NameError:
            print("AttentionUNet not defined, falling back to standard UNet")
            model = UNet(n_channels=3, n_classes=1, bilinear=False)
    else:
        model = UNet(n_channels=3, n_classes=1, bilinear=False)
        print("Using standard UNet model")
    
    # Self-supervised pretraining if requested
    if args.use_pretrained:
        print(f"Performing self-supervised pretraining with {args.pretrain_method} method")
        pretrain_dir = os.path.join(checkpoint_dir, "pretrained")
        os.makedirs(pretrain_dir, exist_ok=True)
        
        # Use all available images for pretraining (both PH2 and Mpox)
        pretrain_images_dir = [args.ph2_dir, args.mpox_dir]
        
        # Extract encoder part (simplified approach)
        # This is a placeholder - in practice you'd need to modify the UNet class
        # or create a wrapper to extract just the encoder
        encoder = model  # Using full model as placeholder
        
        # Pretrain the model
        pretrain_with_self_supervision(
            encoder, pretrain_images_dir, pretrain_dir,
            batch_size=args.batch_size, num_epochs=args.pretrain_epochs,
            learning_rate=args.lr/10, method=args.pretrain_method, device=device
        )
        
        print("Self-supervised pretraining completed")
    
    model.to(device)
    
    # Create optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Create criterion
    criterion = EnhancedCombinedLoss(
        bce_weight=0.3, dice_weight=0.5, focal_weight=0.2, reg_weight=0.0005
    )
    print("Using EnhancedCombinedLoss")
    
    # Learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
        print("Using CosineAnnealingLR scheduler")
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5, verbose=True
        )
        print("Using ReduceLROnPlateau scheduler")
    
    # Train the model with cross-dataset evaluation
    history = train_model(
        model, train_loader, val_loader, test_loader, 
        criterion, optimizer, scheduler,
        args.epochs, device, checkpoint_dir, log_interval=5,
        early_stopping_patience=15
    )
    
    # Evaluate final model on cross-dataset (Mpox)
    print("\nEvaluating final model on cross-dataset (Mpox)...")
    test_metrics = evaluate_clinical_metrics(
        model, test_loader, threshold=args.threshold, device=device
    )
    print(f"Final Cross-Dataset Metrics:")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    
    # Save final clinical metrics
    with open(os.path.join(checkpoint_dir, "final_cross_dataset_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)
    
    # Visualize predictions on cross-dataset
    print("Generating prediction visualizations on cross-dataset...")
    vis_dir = os.path.join(checkpoint_dir, "cross_dataset_visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    visualize_predictions(model, test_loader, device, num_samples=10, output_dir=vis_dir)
    
    print(f"Cross-dataset training completed. Results saved to {checkpoint_dir}")


if __name__ == "__main__":
    main()
