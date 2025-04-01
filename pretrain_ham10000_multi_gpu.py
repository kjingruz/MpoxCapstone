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

# Import our model and HAM10000 loader
from unet_model import UNet
from unet_encoder import UNetEncoder
try:
    from attention_unet import AttentionUNet
except ImportError:
    print("AttentionUNet not found, will use standard UNet")
    AttentionUNet = None

from ham10000_loader import HAM10000ContrastiveDataset

class SimSiam(nn.Module):
    """
    SimSiam model for self-supervised learning
    """
    def __init__(self, base_encoder, dim=512, pred_dim=128):
        """
        Args:
            base_encoder: backbone model (UNet encoder)
            dim: feature dimension (512 by default)
            pred_dim: hidden dimension of the predictor (128 by default)
        """
        super(SimSiam, self).__init__()
        
        # Create encoder (f) - use the provided base encoder
        self.encoder = base_encoder
        
        # Add a projection MLP (g)
        # Note: UNet bottleneck is usually 1024 channels for base UNet with bilinear=False
        # Adjust in_channels based on your UNet implementation
        self.projector = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),  # No spatial change, just channel projection
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            # Add global average pooling to get a vector
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Add a prediction MLP (h) - works on flattened vectors
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, dim)
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
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
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
        
        # Compute projections - results in (B, dim, 1, 1)
        z1 = self.projector(f1)
        z2 = self.projector(f2)
        
        # Reshape to (B, dim) for the predictor
        z1 = z1.view(z1.size(0), -1)
        z2 = z2.view(z2.size(0), -1)
        
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


def train_simsiam(model, train_loader, optimizer, device, epoch, local_rank, log_interval=10):
    """
    Train SimSiam model for one epoch
    
    Args:
        model: SimSiam model
        train_loader: data loader for training
        optimizer: optimizer
        device: device to use
        epoch: current epoch number
        local_rank: local GPU rank for distributed training
        log_interval: how often to log progress
    
    Returns:
        avg_loss: average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    # Only show progress bar on main process
    if local_rank == 0:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    else:
        pbar = train_loader
    
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
        
        # Update progress bar on main process
        if local_rank == 0 and batch_idx % log_interval == 0:
            pbar.set_postfix({"Loss": total_loss / (batch_idx + 1)})
    
    # Calculate average loss across all processes
    avg_loss = torch.tensor([total_loss / len(train_loader)], device=device)
    if dist.is_initialized():
        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
        avg_loss /= dist.get_world_size()
    
    return avg_loss.item()


def save_checkpoint(model, optimizer, epoch, loss, filename, local_rank=0):
    """
    Save model checkpoint
    
    Args:
        model: model to save
        optimizer: optimizer state
        epoch: current epoch
        loss: current loss
        filename: filename to save to
        local_rank: local GPU rank (only save from rank 0)
    """
    # Only save checkpoint from main process
    if local_rank != 0:
        return
    
    # Unwrap DDP model
    if isinstance(model, DDP):
        model_to_save = model.module
    else:
        model_to_save = model
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_to_save.encoder.state_dict(),  # Save only the encoder part
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")


def setup_distributed(rank, world_size):
    """
    Initialize distributed training
    
    Args:
        rank: Current process rank
        world_size: Total number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl',  # Use NCCL backend for GPU training
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    # Set device for current process
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed training resources"""
    if dist.is_initialized():
        dist.destroy_process_group()


def create_distributed_loader(dataset, batch_size, num_workers, world_size, rank):
    """Create a distributed data loader"""
    # Create a distributed sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )
    
    # Create data loader with sampler
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return loader, sampler


def train_process(rank, world_size, args):
    """
    Main training process function for distributed training
    
    Args:
        rank: Current process rank
        world_size: Total number of processes
        args: Command line arguments
    """
    # Set up distributed training
    setup_distributed(rank, world_size)
    
    # Set device for current process
    device = torch.device(f"cuda:{rank}")
    
    # Create output directory (from rank 0 only)
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(args.output_dir, f"simsiam_run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Save arguments
        with open(os.path.join(run_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(args.output_dir, f"simsiam_run_{timestamp}")
    
    # Synchronize processes to ensure run_dir is created
    if world_size > 1:
        torch.distributed.barrier()
    
    # Create UNet model based on argument
    if args.model_type == 'attention' and AttentionUNet is not None:
        try:
            base_model = AttentionUNet(n_channels=3, n_classes=1, bilinear=False)
            if rank == 0:
                print("Using AttentionUNet model")
        except Exception as e:
            if rank == 0:
                print(f"Error loading AttentionUNet: {e}")
                print("Falling back to standard UNet")
            base_model = UNet(n_channels=3, n_classes=1, bilinear=False)
    else:
        base_model = UNet(n_channels=3, n_classes=1, bilinear=False)
        if rank == 0:
            print("Using standard UNet model")
    
    # Create encoder from UNet
    encoder = UNetEncoder(base_model)
    
    # Create SimSiam model
    model = SimSiam(encoder)
    model = model.to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    
    # Create dataset for contrastive learning
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    # Create strong augmentations for contrastive learning
    transform = A.Compose([
        # Use size parameter for compatibility
        A.RandomResizedCrop(size=(args.img_size, args.img_size), scale=(0.2, 1.0)),
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
        images_dir=args.ham10000_dir,
        transform=transform,
        target_size=(args.img_size, args.img_size)
    )
    
    # Create distributed data loader
    train_loader, train_sampler = create_distributed_loader(
        dataset, args.batch_size, args.num_workers, world_size, rank
    )
    
    if rank == 0:
        print(f"Training with {len(dataset)} images from HAM10000 dataset")
        print(f"Using {world_size} GPUs")
    
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
    
    # Training history (only on rank 0)
    if rank == 0:
        history = {
            'loss': []
        }
    
    # Train the model
    for epoch in range(args.epochs):
        # Set epoch for sampler
        train_sampler.set_epoch(epoch)
        
        # Train for one epoch
        loss = train_simsiam(model, train_loader, optimizer, device, epoch, rank)
        
        # Update learning rate
        scheduler.step()
        
        # Update history (only on rank 0)
        if rank == 0:
            history['loss'].append(loss)
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            save_checkpoint(
                model, optimizer, epoch, loss,
                os.path.join(run_dir, f"simsiam_ep{epoch+1}.pth"),
                rank
            )
    
    # Save final model (encoder only) - only on rank 0
    if rank == 0:
        encoder_path = os.path.join(run_dir, "encoder_final.pth")
        torch.save(model.module.encoder.state_dict(), encoder_path)
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
    
    # Clean up distributed training
    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description='Self-supervised pretraining with HAM10000')
    parser.add_argument('--ham10000_dir', required=True, help='Directory containing HAM10000 images')
    parser.add_argument('--output_dir', default='./pretrained', help='Output directory for pretrained models')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    parser.add_argument('--model_type', default='attention', choices=['attention', 'standard'], 
                       help='UNet model type to use (attention or standard)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--save_interval', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--world_size', type=int, default=None, 
                       help='Number of GPUs to use (default: all available)')
    
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
