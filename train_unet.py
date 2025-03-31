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
import cv2  # Added import for connected components analysis

# Import our model and dataset
from unet_model import UNet
from mpox_dataset import get_data_loaders, MpoxDataset

# Import the attention UNet model if available, otherwise use standard UNet
try:
    from attention_unet import AttentionUNet
    print("Using AttentionUNet model")
except ImportError:
    print("AttentionUNet model not found, will use standard UNet instead")
    AttentionUNet = UNet  # Fallback to standard UNet if AttentionUNet is not available

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

# Combined loss for better performance
class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        
    def forward(self, logits, targets):
        # Ensure targets has the same shape as logits
        if len(targets.shape) == 3:
            targets = targets.unsqueeze(1)
        
        # Ensure targets is a float tensor
        targets = targets.float()
        
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

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
    """Calculate clinical metrics beyond just IoU"""
    model.eval()
    
    total_images = 0
    total_tp = 0  # True positives (component-wise)
    total_fp = 0  # False positives (component-wise)
    total_fn = 0  # False negatives (component-wise)
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Ensure masks have channel dimension and correct type
            if len(masks.shape) == 3:
                masks = masks.unsqueeze(1)
            
            # Forward pass
            outputs = model(images)
            
            # Convert to binary predictions
            preds = (torch.sigmoid(outputs) > threshold).float()
            
            # Convert to numpy for connected component analysis
            batch_size = images.shape[0]
            for i in range(batch_size):
                pred = preds[i, 0].cpu().numpy()
                mask = masks[i, 0].cpu().numpy()
                
                # Connected component analysis for prediction
                pred_labels, pred_count = cv2.connectedComponents(
                    (pred > 0).astype(np.uint8)
                )
                
                # Connected component analysis for ground truth
                gt_labels, gt_count = cv2.connectedComponents(
                    (mask > 0).astype(np.uint8)
                )
                
                # Count matches (true positives)
                tp = 0
                matched_pred = set()
                
                for gt_idx in range(1, gt_count):  # Skip background (0)
                    gt_mask = (gt_labels == gt_idx)
                    best_iou = 0
                    best_pred = -1
                    
                    for pred_idx in range(1, pred_count):
                        if pred_idx in matched_pred:
                            continue
                            
                        pred_mask = (pred_labels == pred_idx)
                        
                        # Calculate IoU for this component pair
                        intersection = np.logical_and(gt_mask, pred_mask).sum()
                        union = np.logical_or(gt_mask, pred_mask).sum()
                        iou = intersection / union if union > 0 else 0
                        
                        if iou > 0.5 and iou > best_iou:  # 0.5 IoU threshold for match
                            best_iou = iou
                            best_pred = pred_idx
                    
                    if best_pred != -1:
                        tp += 1
                        matched_pred.add(best_pred)
                
                # False positives: predicted components that didn't match any ground truth
                fp = pred_count - 1 - len(matched_pred)  # -1 for background
                
                # False negatives: ground truth components that weren't matched
                fn = gt_count - 1 - tp  # -1 for background
                
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_images += 1
    
    # Calculate precision, recall, F1
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Average lesions per image
    avg_lesions_true = (total_tp + total_fn) / total_images if total_images > 0 else 0
    avg_lesions_pred = (total_tp + total_fp) / total_images if total_images > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_lesions_true': avg_lesions_true,
        'avg_lesions_pred': avg_lesions_pred,
        'false_positive_rate': total_fp / total_images if total_images > 0 else 0
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

# Main training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs, device, checkpoint_dir, log_interval=1,
                early_stopping_patience=15):  # Increased patience
    # Initialize training history
    history = {
        'train_loss': [],
        'train_iou': [],
        'val_loss': [],
        'val_iou': [],
        'clinical_metrics': []  # Added clinical metrics tracking
    }
    
    # Early stopping variables
    best_val_iou = 0
    best_f1_score = 0  # Added tracking for best F1 score
    patience_counter = 0
    
    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Start training
    print("Starting training...")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train for one epoch
        train_loss, train_iou = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_iou = validate(model, val_loader, criterion, device)
        
        # Calculate clinical metrics every 5 epochs to avoid slowing down training
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            clinical_metrics = evaluate_clinical_metrics(model, val_loader, threshold=0.7, device=device)
            history['clinical_metrics'].append(clinical_metrics)
            
            print(f"Clinical Metrics - F1: {clinical_metrics['f1']:.4f}, "
                  f"Precision: {clinical_metrics['precision']:.4f}, "
                  f"Recall: {clinical_metrics['recall']:.4f}")
            print(f"Avg True Lesions: {clinical_metrics['avg_lesions_true']:.2f}, "
                  f"Avg Predicted Lesions: {clinical_metrics['avg_lesions_pred']:.2f}")
        else:
            history['clinical_metrics'].append(None)
        
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
        
        # Also check F1 score if clinical metrics were calculated
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            if clinical_metrics['f1'] > best_f1_score:
                best_f1_score = clinical_metrics['f1']
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
                'clinical_metrics': clinical_metrics if epoch % 5 == 0 or epoch == num_epochs - 1 else None,
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
                'clinical_metrics': clinical_metrics if epoch % 5 == 0 or epoch == num_epochs - 1 else None,
                'history': history
            }, checkpoint_path)
    
    # Plot training history
    plot_path = os.path.join(checkpoint_dir, "training_history.png")
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history['train_iou'], label='Train')
    plt.plot(history['val_iou'], label='Validation')
    plt.title('IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    
    # Plot F1 scores if available
    plt.subplot(1, 3, 3)
    f1_scores = [m['f1'] if m is not None else None for m in history['clinical_metrics']]
    valid_epochs = [i for i, score in enumerate(f1_scores) if score is not None]
    valid_scores = [score for score in f1_scores if score is not None]
    
    if valid_scores:
        plt.plot(valid_epochs, valid_scores, marker='o', linestyle='-', color='green')
        plt.title('F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
    
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    return history


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train U-Net for Mpox Lesion Segmentation')
    parser.add_argument('--images_dir', required=True, help='Directory containing images')
    parser.add_argument('--masks_dir', help='Directory containing mask annotations (optional)')
    parser.add_argument('--output_dir', default='./outputs', help='Output directory for models and results')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (increased from 8)')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs (increased from 50)')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate (decreased from 0.001)')
    parser.add_argument('--img_size', type=int, default=256, help='Image size (square)')
    parser.add_argument('--use_pseudo_labels', action='store_true', help='Generate pseudo labels if no masks')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--model_type', default='attention', choices=['attention', 'standard'], 
                       help='UNet model type to use (attention or standard)')
    parser.add_argument('--scheduler', default='cosine', choices=['cosine', 'plateau'],
                       help='Learning rate scheduler to use')
    parser.add_argument('--threshold', type=float, default=0.7, 
                       help='Confidence threshold for predictions (increased from 0.5)')
    
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
    checkpoint_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Log arguments
    with open(os.path.join(checkpoint_dir, "args.txt"), "w") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    # Create data loaders - check if get_data_loaders returns positive class weight
    try:
        # Try with the upgraded function that returns pos_weight
        train_loader, val_loader, pos_weight = get_data_loaders(
            args.images_dir,
            args.masks_dir,
            batch_size=args.batch_size,
            val_split=0.2,
            use_pseudo_labels=args.use_pseudo_labels,
            target_size=(args.img_size, args.img_size),
            num_workers=args.num_workers
        )
        print(f"Using positive class weight: {pos_weight}")
    except ValueError:
        # Fall back to the original function
        train_loader, val_loader = get_data_loaders(
            args.images_dir,
            args.masks_dir,
            batch_size=args.batch_size,
            val_split=0.2,
            use_pseudo_labels=args.use_pseudo_labels,
            target_size=(args.img_size, args.img_size),
            num_workers=args.num_workers
        )
        pos_weight = None
    
    print(f"Training with {len(train_loader.dataset)} images")
    print(f"Validating with {len(val_loader.dataset)} images")
    
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
        
    model.to(device)
    
    # Create optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Create criterion with or without positive class weight
    if pos_weight is not None:
        criterion = EnhancedCombinedLoss(
            bce_weight=0.3, dice_weight=0.5, focal_weight=0.2, reg_weight=0.0005
        )
        print("Using EnhancedCombinedLoss with class weighting")
    else:
        criterion = EnhancedCombinedLoss(
            bce_weight=0.3, dice_weight=0.5, focal_weight=0.2, reg_weight=0.0005
        )
        print("Using EnhancedCombinedLoss without class weighting")
    
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
    
    # Train the model
    history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        args.epochs, device, checkpoint_dir, log_interval=5,
        early_stopping_patience=15  # Increased from 10
    )
    
    # Evaluate final model clinical metrics
    print("\nEvaluating final model clinical metrics...")
    clinical_metrics = evaluate_clinical_metrics(
        model, val_loader, threshold=args.threshold, device=device
    )
    print(f"Final Clinical Metrics:")
    print(f"F1 Score: {clinical_metrics['f1']:.4f}")
    print(f"Precision: {clinical_metrics['precision']:.4f}")
    print(f"Recall: {clinical_metrics['recall']:.4f}")
    print(f"Avg True Lesions: {clinical_metrics['avg_lesions_true']:.2f}")
    print(f"Avg Predicted Lesions: {clinical_metrics['avg_lesions_pred']:.2f}")
    print(f"False Positive Rate: {clinical_metrics['false_positive_rate']:.4f}")
    
    # Save final clinical metrics
    with open(os.path.join(checkpoint_dir, "final_clinical_metrics.txt"), "w") as f:
        for metric, value in clinical_metrics.items():
            f.write(f"{metric}: {value}\n")
    
    # Visualize predictions on validation set
    print("Generating prediction visualizations...")
    vis_dir = os.path.join(checkpoint_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    visualize_predictions(model, val_loader, device, num_samples=10, output_dir=vis_dir)
    
    print(f"Training completed. Results saved to {checkpoint_dir}")


if __name__ == "__main__":
    main()
