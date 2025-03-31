#!/bin/bash

###############################################################################
# Slurm Job Configuration
###############################################################################
#SBATCH --job-name=Mpox_UNet
#SBATCH --output=Mpox_UNet_%j.log
#SBATCH --error=Mpox_UNet_%j.log
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=main

###############################################################################
# 1) HPC Modules
###############################################################################
echo "Loading HPC modules..."
module load python/3.10
module load cuda/11.8.0
module load cudnn/8.6.0

# Print loaded modules for verification
module list

###############################################################################
# 2) Setup directories
###############################################################################
VENV_DIR="$HOME/mpox_venv"
SCRIPT_DIR="$(pwd)"
WEIGHTS_DIR="$SCRIPT_DIR/model_weights"
RESULTS_DIR="$SCRIPT_DIR/results"
VISUALIZATION_DIR="$SCRIPT_DIR/visualizations"

mkdir -p "$WEIGHTS_DIR"
mkdir -p "$RESULTS_DIR"
mkdir -p "$VISUALIZATION_DIR"

###############################################################################
# 3) Check for GPU
###############################################################################
echo "Checking for GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "===== GPU status before job starts ====="
    nvidia-smi
    echo "========================================="
    GPU_FLAG=""
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/' 2>/dev/null)
    echo "Detected CUDA Version: $CUDA_VERSION"
else
    echo "WARNING: No GPU detected, using CPU only"
    echo "This is not recommended for training. Check if CUDA modules are loaded."
    GPU_FLAG="--no_cuda"
fi

###############################################################################
# 4) Virtual Environment Setup
###############################################################################
# Remove existing venv to avoid compatibility issues
if [ -d "$VENV_DIR" ]; then
  echo "Removing existing virtual environment to ensure clean setup"
  rm -rf "$VENV_DIR"
fi

echo "Creating new virtual environment in $VENV_DIR"
python3 -m venv "$VENV_DIR"

echo "Activating virtual environment"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel

###############################################################################
# 5) Install Required Libraries WITH CORRECT NUMPY VERSION
###############################################################################
echo "Installing required Python packages with compatible versions..."

# IMPORTANT: Install NumPy 1.x first to avoid compatibility issues with OpenCV
echo "Installing NumPy 1.x for OpenCV compatibility"
pip install "numpy<2.0.0"

# Check NumPy version
NUMPY_VERSION=$(python -c "import numpy; print(numpy.__version__)")
echo "Installed NumPy version: $NUMPY_VERSION"

# Install PyTorch with CUDA support if GPU is available
if [ -z "$GPU_FLAG" ]; then
    # Install PyTorch with CUDA 11.8 support (specific to loaded module)
    echo "Installing PyTorch with CUDA 11.8 support"
    pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
else
    echo "Installing PyTorch for CPU"
    pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
fi

# Install other required packages
pip install pandas matplotlib scikit-learn tqdm
pip install Pillow

###############################################################################
# 6) Install Headless OpenCV AFTER NumPy is installed
###############################################################################
echo "Setting up headless OpenCV..."

# First uninstall any existing OpenCV packages
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless

# Install the headless version of OpenCV (specific version for compatibility)
pip install opencv-python-headless==4.7.0.72

# Verify installation
echo "Verifying OpenCV installation..."
python -c "import cv2; print('OpenCV version:', cv2.__version__)" || {
    echo "ERROR: Failed to verify OpenCV installation"
    echo "Trying alternative OpenCV version..."
    pip uninstall -y opencv-python-headless
    pip install opencv-python-headless==4.6.0.66
    python -c "import cv2; print('OpenCV version:', cv2.__version__)" || {
        echo "ERROR: Still unable to load OpenCV. Exiting."
        exit 1
    }
}

# Install albumentations after OpenCV
pip install albumentations

###############################################################################
# 7) Configure Matplotlib for Non-Interactive Backend
###############################################################################
# Create matplotlibrc file to use Agg backend (non-interactive)
mkdir -p ~/.config/matplotlib
echo "backend: Agg" > ~/.config/matplotlib/matplotlibrc

# Verify matplotlib configuration
python -c "import matplotlib; print('Matplotlib backend:', matplotlib.get_backend())" || {
    echo "WARNING: Failed to verify matplotlib configuration"
}

###############################################################################
# 8) Fix the dataset boolean mask issue
###############################################################################
echo "Patching the dataset code to fix the boolean mask issue..."

# Create a temporary file with the fix
cat > mpox_dataset_fix.py << 'EOL'
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
from pathlib import Path
import random


class MpoxDataset(Dataset):
    """Mpox lesion segmentation dataset"""

    def __init__(self, images_dir, masks_dir=None, transform=None, target_size=(256, 256),
                 use_pseudo_labels=False, aug_transform=None):
        """
        Args:
            images_dir (str): Directory with the mpox images
            masks_dir (str, optional): Directory with the mask images. If None and use_pseudo_labels is True,
                                      pseudo labels will be generated.
            transform (callable, optional): Optional transform to be applied on a sample
            target_size (tuple): Resize images to this size
            use_pseudo_labels (bool): Whether to generate pseudo labels if masks_dir is None
            aug_transform (callable, optional): Data augmentation transforms
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir) if masks_dir else None
        self.transform = transform
        self.aug_transform = aug_transform
        self.target_size = target_size
        self.use_pseudo_labels = use_pseudo_labels

        # Find all image files
        self.image_files = sorted([f for f in os.listdir(images_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Get image path
        img_path = os.path.join(self.images_dir, self.image_files[idx])

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Resize to target size
        image = image.resize(self.target_size, Image.LANCZOS)

        # Convert to numpy for processing if needed
        image_np = np.array(image)

        # Load or generate mask
        if self.masks_dir:
            # Try to find corresponding mask
            mask_filename = self.image_files[idx].replace('.jpg', '_mask.png').replace('.jpeg', '_mask.png')
            mask_path = os.path.join(self.masks_dir, mask_filename)

            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')
                mask = mask.resize(self.target_size, Image.NEAREST)
                mask = np.array(mask) > 0  # Convert to binary
            else:
                # No mask found, create empty mask
                mask = np.zeros(self.target_size, dtype=np.uint8)
        elif self.use_pseudo_labels:
            # Generate pseudo label using our previous detection approach
            mask = self._generate_pseudo_mask(image_np)
        else:
            # No mask, create empty mask
            mask = np.zeros(self.target_size, dtype=np.uint8)

        # IMPORTANT FIX: Convert boolean mask to uint8 before augmentation
        # This prevents the OpenCV error with boolean arrays
        if isinstance(mask, np.ndarray) and mask.dtype == bool:
            mask = mask.astype(np.uint8) * 255

        # Apply data augmentation if provided
        if self.aug_transform:
            augmented = self.aug_transform(image=image_np, mask=mask)
            image_np = augmented['image']
            mask = augmented['mask']

            # Convert mask back to boolean after augmentation if needed
            if isinstance(mask, np.ndarray) and mask.dtype != bool:
                mask = mask > 0

        # Apply base transforms (normalization, etc.)
        if self.transform:
            image_np = self.transform(image_np)
        else:
            # Default normalization - check if already a tensor (from augmentation)
            if not isinstance(image_np, torch.Tensor):
                # For numpy arrays, normalize and convert to tensor
                image_np = image_np / 255.0
                image_np = torch.from_numpy(image_np.transpose(2, 0, 1)).float()

        # Convert mask to tensor
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask.astype(np.float32))
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)  # Add channel dimension

        return {
            'image': image_np,
            'mask': mask,
            'filename': self.image_files[idx]
        }

    def _generate_pseudo_mask(self, image):
        """Generate a pseudo mask using traditional CV methods"""
        # Convert to HSV color space for better color-based segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Extract value channel (brightness)
        _, _, v_channel = cv2.split(hsv)

        # Create a darkness map to identify darker regions
        # First blur to get average skin tone
        blurred_v = cv2.GaussianBlur(v_channel, (51, 51), 0)
        # Subtract to find regions darker than surroundings
        darkness_map = cv2.subtract(blurred_v, v_channel)

        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        darkness_enhanced = clahe.apply(darkness_map)

        # Threshold to identify darker regions
        _, binary = cv2.threshold(darkness_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Convert to uint8 mask instead of boolean to avoid OpenCV issues
        mask = (cleaned > 0).astype(np.uint8)

        return mask


def get_data_loaders(images_dir, masks_dir=None, batch_size=8, val_split=0.2,
                    use_pseudo_labels=False, target_size=(256, 256), num_workers=4):
    """
    Create train and validation data loaders

    Args:
        images_dir (str): Directory with images
        masks_dir (str, optional): Directory with mask images
        batch_size (int): Batch size for data loaders
        val_split (float): Validation split ratio (0-1)
        use_pseudo_labels (bool): Whether to use pseudo labels if no masks provided
        target_size (tuple): Target image size
        num_workers (int): Number of workers for data loading

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Create list of image files
    all_images = sorted([f for f in os.listdir(images_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    # Split into train and validation
    indices = list(range(len(all_images)))
    random.seed(42)  # For reproducibility
    random.shuffle(indices)

    val_size = int(len(indices) * val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    train_images = [all_images[i] for i in train_indices]
    val_images = [all_images[i] for i in val_indices]

    # Create train and validation directories
    train_dir = os.path.join(images_dir, '../train')
    val_dir = os.path.join(images_dir, '../val')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Create symlinks to original images
    for img in train_images:
        src = os.path.join(images_dir, img)
        dst = os.path.join(train_dir, img)
        if not os.path.exists(dst):
            if os.name == 'nt':  # Windows
                import shutil
                shutil.copy2(src, dst)
            else:  # Unix
                os.symlink(src, dst)

    for img in val_images:
        src = os.path.join(images_dir, img)
        dst = os.path.join(val_dir, img)
        if not os.path.exists(dst):
            if os.name == 'nt':  # Windows
                import shutil
                shutil.copy2(src, dst)
            else:  # Unix
                os.symlink(src, dst)

    # Create datasets
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        # Data augmentation for training
        train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussianBlur(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        # Only normalization for validation
        val_transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        train_dataset = MpoxDataset(
            train_dir, masks_dir, transform=None,
            target_size=target_size, use_pseudo_labels=use_pseudo_labels,
            aug_transform=train_transform
        )

        val_dataset = MpoxDataset(
            val_dir, masks_dir, transform=None,
            target_size=target_size, use_pseudo_labels=use_pseudo_labels,
            aug_transform=val_transform
        )
    except ImportError:
        # Fallback if albumentations is not available
        print("Albumentations not available, using basic transforms")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_dataset = MpoxDataset(
            train_dir, masks_dir, transform=transform,
            target_size=target_size, use_pseudo_labels=use_pseudo_labels
        )

        val_dataset = MpoxDataset(
            val_dir, masks_dir, transform=transform,
            target_size=target_size, use_pseudo_labels=use_pseudo_labels
        )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader
EOL

# Replace the original dataset file with our fixed version
cp mpox_dataset_fix.py mpox_dataset.py

###############################################################################
# 9) Training parameters
###############################################################################
SEED=42
MAX_EPOCHS=50
BATCH_SIZE=16
LEARNING_RATE=0.001
IMAGE_SIZE=256
USE_PSEUDO_LABELS=true
# Reduce number of workers to avoid potential DataLoader issues
NUM_WORKERS=2

###############################################################################
# 10) Verify CUDA is available in Python
###############################################################################
echo "Verifying CUDA is available to PyTorch..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')" || {
    echo "WARNING: Unable to verify CUDA availability. Check PyTorch installation."
}

###############################################################################
# 11) Run training with better error handling
###############################################################################
echo "Starting U-Net training..."

# Run training with specified parameters and redirect output for better logging
python train_unet.py \
    --images_dir "$SCRIPT_DIR/data/Monkey_Pox" \
    --output_dir "$WEIGHTS_DIR" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$MAX_EPOCHS" \
    --lr "$LEARNING_RATE" \
    --img_size "$IMAGE_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --use_pseudo_labels \
    $GPU_FLAG 2>&1 | tee training_log.txt

# Check if training was successful
if [ $? -ne 0 ]; then
    echo "ERROR: Training failed. See training_log.txt for details."
    # Try to give more diagnostic information
    echo "Last few lines of training log:"
    tail -n 20 training_log.txt
    exit 1
fi

###############################################################################
# 12) Find best model
###############################################################################
BEST_MODEL=$(find "$WEIGHTS_DIR" -name "best_model.pth" | head -1)

if [ -z "$BEST_MODEL" ]; then
    echo "WARNING: No best model found! Checking for any checkpoint..."
    BEST_MODEL=$(find "$WEIGHTS_DIR" -name "*.pth" | head -1)

    if [ -z "$BEST_MODEL" ]; then
        echo "ERROR: No model checkpoints found. Training may have failed."
        exit 1
    fi
fi

echo "Best model found: $BEST_MODEL"

###############################################################################
# 13) Run inference
###############################################################################
echo "Running inference on Mpox data..."

python inference.py \
    --model "$BEST_MODEL" \
    --input "$SCRIPT_DIR/data/Monkey_Pox" \
    --output "$RESULTS_DIR" \
    --img_size "$IMAGE_SIZE" \
    $GPU_FLAG 2>&1 | tee inference_log.txt

# Check if inference was successful
if [ $? -ne 0 ]; then
    echo "ERROR: Inference failed. See inference_log.txt for details."
    exit 1
fi

echo "============================================================="
echo "JOB COMPLETED SUCCESSFULLY"
echo "============================================================="
echo "Results saved to: $RESULTS_DIR"
echo "Best model saved to: $BEST_MODEL"
echo "============================================================="

# Display GPU status after job completion
if command -v nvidia-smi &> /dev/null; then
    echo "===== GPU status after job completion ====="
    nvidia-smi
    echo "=========================================="
fi
