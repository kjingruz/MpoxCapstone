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

# Import our models and datasets
from unet_model import UNet
from attention_unet import AttentionUNet
from cross_dataset_loader import create_cross_dataset_loaders
from unet_encoder import UNetEncoder

# Import loss functions and evaluation metrics from previous scripts
from train_cross_dataset import DiceLoss, FocalLoss, EnhancedCombinedLoss, iou_score, evaluate_clinical_metrics

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
    if model_type == 'attention':
        try:
            base_model = AttentionUNet(n_channels=3, n_classes=1, bilinear=False)
            print("Using AttentionUNet model")
        except Exception as e:
            print(f"Error loading AttentionUNet: {e}")
            print("Falling back to standard UNet")
            base_model = UNet(n_channels=3, n_classes=1, bilinear=False)
    else:
        base_model = UNet(n_channels=3, n_classes=1, bilinear=False)
        print("Using standard UNet model")
    
    # Create encoder from base model
    encoder = UNetEncoder(base_model)
    
    # Check if file exists
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained weights file not found: {pretrained_path}")
    
    # Load pretrained weights
    checkpoint = torch.load(pretrained_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        # Load from model checkpoint
        encoder.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Direct state dict
        encoder.load_state_dict(checkpoint)
    
    print(f"Loaded pretrained encoder from {pretrained_path}")
    return encoder


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    epoch_iou = 0
    
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


def validate(model, dataloader, criterion, device):
    """Validate model"""
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
            
            # Calculate loss
            loss = criterion(outputs, masks, model)
            
            # Calculate IoU
            batch_iou = iou_score(outputs, masks)
            
            # Update metrics
            val_loss += loss.item()
            val_iou += batch_iou.item()
    
    # Return average loss and IoU
    return val_loss / len(dataloader), val_iou / len(dataloader)


def visualize_predictions(model, dataloader, device, num_samples=3, output_dir=None):
    """Visualize model predictions"""
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
                    mask = masks[j, 0].cpu().numpy()
                else:
                    mask = masks[j].cpu().numpy()
                
                pred = preds[j, 0].cpu().numpy()
                
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


def train_with_transfer_learning(model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler,
                                num_epochs, device, checkpoint_dir, fine_tune_epoch=50, log_interval=5,
                                early_stopping_patience=15):
    """Train model with transfer learning approach"""
    
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
    print("Starting training with transfer learning...")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Unfreeze encoder for fine-tuning after specified epoch
        if epoch == fine_tune_epoch:
            print("Unfreezing encoder for fine-tuning...")
            if hasattr(model, 'unfreeze_encoder'):
                model.unfreeze_encoder()
                
                # Adjust learning rate for fine-tuning
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] / 10
                
                print(f"Adjusted learning rate to {optimizer.param_groups[0]['lr']}")
        
        # Train for one epoch
        train_loss, train_iou = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_iou = validate(model, val_loader, criterion, device)
        
        # Calculate validation metrics every 5 epochs
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            val_metrics = evaluate_clinical_metrics(model, val_loader, threshold=0.7, device=device)
            history['val_metrics'].append(val_metrics)
            
            # Also evaluate on test set (Mpox)
            test_metrics = evaluate_clinical_metrics(model, test_loader, threshold=0.7, device=device)
            history['test_metrics'].append(test_metrics)
            
            print(f"Val Metrics - F1: {val_metrics['f1']:.4f}, "
                  f"Precision: {val_metrics['precision']:.4f}, "
                  f"Recall: {val_metrics['recall']:.4f}")
            
            print(f"Test Metrics (Mpox) - F1: {test_metrics['f1']:.4f}, "
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
    plot_history(history, checkpoint_dir)
    
    return history


def plot_history(history, output_dir):
    """Plot training history metrics"""
    
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


def main():
    parser = argparse.ArgumentParser(description='Transfer learning with HAM10000 pretraining and PH2 to Mpox')
    parser.add_argument('--ph2_dir', required=True, help='Directory containing PH2 dataset')
    parser.add_argument('--mpox_dir', required=True, help='Directory containing Mpox dataset')
    parser.add_argument('--pretrained_weights', required=True, help='Path to pretrained encoder weights')
    parser.add_argument('--output_dir', default='./transfer_learning_outputs', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    parser.add_argument('--fine_tune_epoch', type=int, default=50, help='Epoch to start fine-tuning encoder')
    parser.add_argument('--model_type', default='attention', choices=['attention', 'standard'], 
                       help='UNet model type to use (attention or standard)')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder during initial training')
    parser.add_argument('--scheduler', default='cosine', choices=['cosine', 'plateau'],
                       help='Learning rate scheduler to use')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    
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
    checkpoint_dir = os.path.join(args.output_dir, f"transfer_learning_run_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Log arguments
    with open(os.path.join(checkpoint_dir, "args.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Create data loaders for cross-dataset training
    train_loader, val_loader, test_loader = create_cross_dataset_loaders(
        args.ph2_dir, args.mpox_dir, batch_size=args.batch_size, 
        target_size=(args.img_size, args.img_size), num_workers=args.num_workers
    )
    
    print(f"Training with {len(train_loader.dataset)} images from PH2 dataset")
    print(f"Validating with {len(val_loader.dataset)} images from PH2 dataset")
    print(f"Testing with {len(test_loader.dataset)} images from Mpox dataset")
    
    # Create base UNet model based on specified type
    if args.model_type == 'attention':
        try:
            base_model = AttentionUNet(n_channels=3, n_classes=1, bilinear=False)
            print("Using AttentionUNet model")
        except Exception as e:
            print(f"Error loading AttentionUNet: {e}")
            print("Falling back to standard UNet")
            base_model = UNet(n_channels=3, n_classes=1, bilinear=False)
    else:
        base_model = UNet(n_channels=3, n_classes=1, bilinear=False)
        print("Using standard UNet model")
    
    # Load pretrained encoder
    pretrained_encoder = load_pretrained_encoder(
        args.pretrained_weights, args.model_type, device
    )
    
    # Create transfer learning model
    model = TransferLearningUNet(
        base_model, pretrained_encoder, freeze_encoder=args.freeze_encoder
    )
    model.to(device)
    
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
        print("Using CosineAnnealingLR scheduler")
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5, verbose=True
        )
        print("Using ReduceLROnPlateau scheduler")
    
    # Train model with transfer learning
    history = train_with_transfer_learning(
        model, train_loader, val_loader, test_loader,
        criterion, optimizer, scheduler,
        args.epochs, device, checkpoint_dir,
        fine_tune_epoch=args.fine_tune_epoch
    )
    
    # Evaluate final model on Mpox dataset
    print("\nEvaluating final model on Mpox dataset...")
    test_metrics = evaluate_clinical_metrics(
        model, test_loader, threshold=0.7, device=device
    )
    print(f"Final Mpox Dataset Metrics:")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    
    # Save final metrics
    with open(os.path.join(checkpoint_dir, "final_mpox_metrics.json"), 'w') as f:
        json.dump(test_metrics, f, indent=4)
    
    # Visualize predictions on Mpox dataset
    print("Generating prediction visualizations on Mpox dataset...")
    vis_dir = os.path.join(checkpoint_dir, "mpox_visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    visualize_predictions(model, test_loader, device, num_samples=10, output_dir=vis_dir)
    
    print(f"Transfer learning completed. Results saved to {checkpoint_dir}")


if __name__ == "__main__":
    main()
