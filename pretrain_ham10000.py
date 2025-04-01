import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import json

# Import our model and HAM10000 loader
from unet_model import UNet
from attention_unet import AttentionUNet
from ham10000_loader import create_ham10000_contrastive_loader

class SimSiam(nn.Module):
    """
    SimSiam model for self-supervised learning
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        Args:
            base_encoder: backbone model (UNet encoder)
            dim: feature dimension (2048 by default)
            pred_dim: hidden dimension of the predictor (512 by default)
        """
        super(SimSiam, self).__init__()
        
        # Create encoder (f) - use the provided base encoder
        self.encoder = base_encoder
        
        # Add a projection MLP (g)
        self.projector = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),  # No spatial change, just channel projection
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
        )
        
        # Add a prediction MLP (h)
        self.predictor = nn.Sequential(
            nn.Conv2d(dim, pred_dim, kernel_size=1),
            nn.BatchNorm2d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(pred_dim, dim, kernel_size=1)
        )
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        # Initialize the projector and predictor with small weights
        for m in self.projector.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        for m in self.predictor.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x1, x2):
        """
        Forward pass through SimSiam model
        
        Args:
            x1: first view of the batch
            x2: second view of the batch
        
        Returns:
            p1, p2: predictions
            z1, z2: projections (targets)
        """
        # Compute features
        f1 = self.encoder(x1)
        f2 = self.encoder(x2)
        
        # Compute projections
        z1 = self.projector(f1)
        z2 = self.projector(f2)
        
        # Compute predictions
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        
        return p1, p2, z1.detach(), z2.detach()


class UNetEncoder(nn.Module):
    """
    Extract encoder part from UNet for self-supervised learning
    """
    def __init__(self, unet_model):
        super(UNetEncoder, self).__init__()
        
        # Extract the encoder part from UNet
        # This depends on the specific UNet implementation
        self.inc = unet_model.inc
        self.down1 = unet_model.down1
        self.down2 = unet_model.down2
        self.down3 = unet_model.down3
        self.down4 = unet_model.down4
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x5  # Return the bottleneck features


def negcos_sim_loss(p, z):
    """
    Negative cosine similarity loss as used in SimSiam
    
    Args:
        p: prediction
        z: target
    
    Returns:
        loss: mean of negative cosine similarity
    """
    z = z.detach()  # Stop gradient
    
    # Normalize the vectors
    p = nn.functional.normalize(p, dim=1)
    z = nn.functional.normalize(z, dim=1)
    
    # Negative cosine similarity
    return -(p * z).sum(dim=1).mean()


def train_simsiam(model, train_loader, optimizer, device, epoch, log_interval=10):
    """
    Train SimSiam model for one epoch
    
    Args:
        model: SimSiam model
        train_loader: data loader for training
        optimizer: optimizer
        device: device to use
        epoch: current epoch number
        log_interval: how often to log progress
    
    Returns:
        avg_loss: average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    with tqdm(train_loader, desc=f"Epoch {epoch+1}") as pbar:
        for batch_idx, batch in enumerate(pbar):
            view1, view2 = batch['view1'].to(device), batch['view2'].to(device)
            
            # Forward pass
            p1, p2, z1, z2 = model(view1, view2)
            
            # Compute loss: SimSiam uses negative cosine similarity
            loss = negcos_sim_loss(p1, z2) / 2 + negcos_sim_loss(p2, z1) / 2
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            
            # Update progress bar
            if batch_idx % log_interval == 0:
                pbar.set_postfix({"Loss": total_loss / (batch_idx + 1)})
    
    return total_loss / len(train_loader)


def save_checkpoint(model, optimizer, epoch, loss, filename):
    """
    Save model checkpoint
    
    Args:
        model: model to save
        optimizer: optimizer state
        epoch: current epoch
        loss: current loss
        filename: filename to save to
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.encoder.state_dict(),  # Save only the encoder part
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description='Self-supervised pretraining with HAM10000')
    parser.add_argument('--ham10000_dir', required=True, help='Directory containing HAM10000 images')
    parser.add_argument('--output_dir', default='./pretrained', help='Output directory for pretrained models')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    parser.add_argument('--model_type', default='attention', choices=['attention', 'standard'], 
                       help='UNet model type to use (attention or standard)')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--save_interval', type=int, default=10, help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"simsiam_run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(run_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Set device
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")
    
    # Create UNet model based on argument
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
    
    # Create encoder from UNet
    encoder = UNetEncoder(base_model)
    
    # Create SimSiam model
    model = SimSiam(encoder)
    model = model.to(device)
    
    # Create contrastive data loader for HAM10000
    train_loader = create_ham10000_contrastive_loader(
        args.ham10000_dir,
        batch_size=args.batch_size,
        target_size=(args.img_size, args.img_size),
        num_workers=args.num_workers
    )
    
    print(f"Training with {len(train_loader.dataset)} images from HAM10000 dataset")
    
    # Create optimizer and scheduler
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=0
    )
    
    # Training history
    history = {
        'loss': []
    }
    
    # Train the model
    for epoch in range(args.epochs):
        # Train for one epoch
        loss = train_simsiam(model, train_loader, optimizer, device, epoch)
        
        # Update learning rate
        scheduler.step()
        
        # Update history
        history['loss'].append(loss)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            save_checkpoint(
                model, optimizer, epoch, loss,
                os.path.join(run_dir, f"simsiam_ep{epoch+1}.pth")
            )
    
    # Save final model (encoder only)
    encoder_path = os.path.join(run_dir, "encoder_final.pth")
    torch.save(encoder.state_dict(), encoder_path)
    print(f"Final encoder saved to {encoder_path}")
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, 'training_loss.png'))
    plt.close()
    
    print(f"Pretraining completed. Results saved to {run_dir}")


if __name__ == "__main__":
    main()
