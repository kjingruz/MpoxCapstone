import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

class RotationDataset(Dataset):
    """Dataset for rotation prediction task (self-supervised learning)"""
    def __init__(self, images_dir, transform=None, target_size=(256, 256)):
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.target_size = target_size
        
        # Find all image files
        self.image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            self.image_files.extend(list(self.images_dir.glob(f'**/*{ext}')))
        
        print(f"Found {len(self.image_files)} images for self-supervised learning")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Resize to target size
        image = image.resize(self.target_size, Image.LANCZOS)
        
        # Convert to numpy array
        image_np = np.array(image)
        
        # Apply transform if provided
        if self.transform:
            augmented = self.transform(image=image_np)
            image_np = augmented['image']
        
        # Randomly rotate image (0, 90, 180, 270 degrees) and create label
        rotation = random.randint(0, 3)
        rotated_image = np.rot90(image_np, rotation) if isinstance(image_np, np.ndarray) else torch.rot90(image_np, rotation, dims=[1, 2])
        
        return {
            'image': rotated_image,
            'label': rotation  # 0: 0°, 1: 90°, 2: 180°, 3: 270°
        }


class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning (SimCLR-like approach)"""
    def __init__(self, images_dir, transform=None, target_size=(256, 256)):
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.target_size = target_size
        
        # Find all image files
        self.image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            self.image_files.extend(list(self.images_dir.glob(f'**/*{ext}')))
        
        print(f"Found {len(self.image_files)} images for contrastive learning")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Resize to target size
        image = image.resize(self.target_size, Image.LANCZOS)
        
        # Convert to numpy array
        image_np = np.array(image)
        
        # Create two different augmentations of the same image
        if self.transform:
            augmented1 = self.transform(image=image_np)
            augmented2 = self.transform(image=image_np)
            
            return {
                'view1': augmented1['image'],
                'view2': augmented2['image']
            }
        else:
            # If no transform, just return the same image twice
            return {
                'view1': image_np,
                'view2': image_np
            }


class SimSiamEncoder(nn.Module):
    """Encoder network for SimSiam approach"""
    def __init__(self, base_encoder, projection_dim=2048, hidden_dim=512):
        super(SimSiamEncoder, self).__init__()
        
        # Use the base encoder (e.g., UNet encoder)
        self.encoder = base_encoder
        
        # Projection MLP
        self.projector = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Prediction MLP
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, hidden_dim)
        )
    
    def forward(self, x):
        # Get features from the encoder
        features = self.encoder(x)
        
        # Apply projector
        z = self.projector(features)
        
        # Apply predictor
        p = self.predictor(z)
        
        return z, p


class MaskedAutoencoder(nn.Module):
    """Simple masked autoencoder for self-supervised learning"""
    def __init__(self, encoder, decoder, mask_ratio=0.75):
        super(MaskedAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask_ratio = mask_ratio
    
    def random_masking(self, x, mask_ratio):
        N, L = x.shape  # batch, length
        len_keep = int(L * (1 - mask_ratio))
        
        # Sort noise for consistent masking across feature dimension
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # Ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep the first len_keep elements
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        
        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward(self, x):
        # Create patch embeddings (simplistic approach)
        batch_size = x.shape[0]
        patches = x.flatten(2).transpose(1, 2)  # [B, L, C]
        
        # Apply random masking
        latent, mask, ids_restore = self.random_masking(patches, self.mask_ratio)
        
        # Encode patches
        encoded = self.encoder(latent)
        
        # Decode patches
        decoded = self.decoder(encoded, ids_restore)
        
        return decoded, mask


def create_self_supervised_transforms():
    """Create transforms for self-supervised learning"""
    # Strong augmentations for contrastive learning
    contrastive_transform = A.Compose([
        A.RandomResizedCrop(height=224, width=224, scale=(0.2, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.ToGray(p=0.2),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Less aggressive transforms for other self-supervised tasks
    standard_transform = A.Compose([
        A.RandomResizedCrop(height=224, width=224),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return contrastive_transform, standard_transform


def train_rotation_prediction(model, train_loader, val_loader, optimizer, num_epochs, device):
    """Train a model using rotation prediction"""
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch in tepoch:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                tepoch.set_postfix(loss=train_loss/len(train_loader), acc=100.*correct/total)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = 100.*correct/total
        print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "rotation_pretrained_model.pth")
            print(f"Model saved with val acc: {val_acc:.2f}%")
    
    return model


def train_contrastive(model, train_loader, optimizer, num_epochs, device, temperature=0.5):
    """Train a model using contrastive learning (SimCLR approach)"""
    model.train()
    
    for epoch in range(num_epochs):
        train_loss = 0
        
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch in tepoch:
                view1 = batch['view1'].to(device)
                view2 = batch['view2'].to(device)
                
                # Get features for both views
                z1 = model(view1)
                z2 = model(view2)
                
                # Normalize embeddings
                z1 = F.normalize(z1, dim=1)
                z2 = F.normalize(z2, dim=1)
                
                # Gather embeddings from all devices
                batch_size = view1.shape[0]
                
                # Create similarity matrix
                representations = torch.cat([z1, z2], dim=0)
                similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
                
                # Remove diagonal (self-similarity)
                similarity_matrix = similarity_matrix[~torch.eye(2*batch_size, dtype=bool, device=similarity_matrix.device)]
                similarity_matrix = similarity_matrix.view(2*batch_size, -1)
                
                # Positives are pairs (i,i+batch_size) and (i+batch_size,i)
                positives = torch.cat([similarity_matrix[range(batch_size), range(batch_size, 2*batch_size)],
                                       similarity_matrix[range(batch_size, 2*batch_size), range(batch_size)]])
                
                # All similarities are negatives except for positives
                negatives = similarity_matrix.view(-1)
                
                # Create labels: positives should have higher similarity
                labels = torch.zeros(2*batch_size, device=device).long()
                
                # Use NT-Xent loss (normalized temperature-scaled cross entropy)
                logits = torch.cat([positives, negatives], dim=0) / temperature
                loss = F.cross_entropy(logits.unsqueeze(0), labels.unsqueeze(0))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                tepoch.set_postfix(loss=train_loss/len(train_loader))
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss/len(train_loader):.4f}")
        
        # Save model checkpoint
        torch.save(model.state_dict(), f"contrastive_pretrained_epoch{epoch+1}.pth")
    
    return model


def pretrain_with_self_supervision(base_model, images_dir, output_dir, batch_size=32, 
                                  num_epochs=50, learning_rate=1e-4, method='simsiam', 
                                  device='cuda'):
    """
    Pretrain the model using self-supervised learning
    
    Args:
        base_model: Base model to pretrain (typically UNet encoder)
        images_dir: Directory containing unlabeled images
        output_dir: Directory to save pretrained model
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        method: Self-supervised method ('rotation', 'contrastive', 'simsiam', 'mae')
        device: Device to use ('cuda' or 'cpu')
    
    Returns:
        Pretrained model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create transforms
    contrastive_transform, standard_transform = create_self_supervised_transforms()
    
    # Set up dataset and loader based on method
    if method == 'rotation':
        dataset = RotationDataset(images_dir, transform=standard_transform)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # Modify model for rotation prediction
        num_features = base_model.fc.in_features if hasattr(base_model, 'fc') else 512
        base_model.fc = nn.Linear(num_features, 4)  # 4 rotation classes
        
        # Train with rotation prediction
        optimizer = optim.Adam(base_model.parameters(), lr=learning_rate)
        pretrained_model = train_rotation_prediction(base_model, train_loader, val_loader, 
                                                    optimizer, num_epochs, device)
    
    elif method == 'contrastive' or method == 'simsiam':
        dataset = ContrastiveDataset(images_dir, transform=contrastive_transform)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        
        # Setup model based on method
        if method == 'simsiam':
            model = SimSiamEncoder(base_model)
        else:
            model = base_model
        
        model = model.to(device)
        
        # Train with contrastive learning
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        pretrained_model = train_contrastive(model, train_loader, optimizer, num_epochs, device)
    
    else:
        raise ValueError(f"Unsupported self-supervised method: {method}")
    
    # Save final model
    final_model_path = os.path.join(output_dir, f"pretrained_{method}_final.pth")
    torch.save(pretrained_model.state_dict(), final_model_path)
    print(f"Final pretrained model saved to {final_model_path}")
    
    return pretrained_model


# Example usage
if __name__ == "__main__":
    # This can be run as a standalone script for pretraining
    import argparse
    from unet_model import UNet
    
    parser = argparse.ArgumentParser(description='Self-supervised pretraining for UNet')
    parser.add_argument('--images_dir', required=True, help='Directory containing images for pretraining')
    parser.add_argument('--output_dir', default='./pretrained_models', help='Output directory for pretrained models')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--method', default='simsiam', choices=['rotation', 'contrastive', 'simsiam'], 
                       help='Self-supervised method to use')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    
    # Create base model
    base_model = UNet(n_channels=3, n_classes=1, bilinear=True)
    
    # Extract encoder part (this is a simplified approach - in practice, you'd need to modify the UNet class)
    encoder = nn.Sequential(*list(base_model.children())[:5])  # Take encoder parts only
    
    # Pretrain
    pretrained_model = pretrain_with_self_supervision(
        encoder, args.images_dir, args.output_dir,
        batch_size=args.batch_size, num_epochs=args.epochs,
        learning_rate=args.lr, method=args.method, device=device
    )
    
    print("Pretraining completed successfully!")
