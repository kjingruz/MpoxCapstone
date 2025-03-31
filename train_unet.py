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

# Main training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs, device, checkpoint_dir, log_interval=1,
                early_stopping_patience=10):
    # Initialize training history
    history = {
        'train_loss': [],
        'train_iou': [],
        'val_loss': [],
        'val_iou': []
    }
    
    # Early stopping variables
    best_val_iou = 0
    patience_counter = 0
    
    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Start training
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train for one epoch
        train_loss, train_iou = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_iou = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        if scheduler:
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
            checkpoint_path = os.path.join(checkpoint_dir, f"best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_iou,
                'history': history
            }, checkpoint_path)
            
            print(f"Saved best model checkpoint (IoU: {val_iou:.4f})")
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
                'history': history
            }, checkpoint_path)
    
    # Plot training history
    plot_path = os.path.join(checkpoint_dir, "training_history.png")
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
    plt.savefig(plot_path)
    plt.close()
    
    return history


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
    
    # Train the model
    history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        args.epochs, device, checkpoint_dir, log_interval=5,
        early_stopping_patience=10
    )
    
    # Visualize predictions on validation set
    print("Generating prediction visualizations...")
    vis_dir = os.path.join(checkpoint_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    visualize_predictions(model, val_loader, device, num_samples=10, output_dir=vis_dir)
    
    print(f"Training completed. Results saved to {checkpoint_dir}")


if __name__ == "__main__":
    main()
