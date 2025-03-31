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
# 8) Check that dataset has the boolean mask fix
###############################################################################
echo "Checking dataset code for boolean mask handling..."
python -c "
import os
print('Current directory:', os.getcwd())
try:
    from mpox_dataset import MpoxDataset
    print('MpoxDataset module loaded successfully')
    
    # Check for WeightedMpoxDataset class
    try:
        from mpox_dataset import WeightedMpoxDataset
        print('WeightedMpoxDataset available - using enhanced dataset')
    except ImportError:
        print('WeightedMpoxDataset not found - using basic dataset')
        
    # Check get_data_loaders return value
    import inspect
    from mpox_dataset import get_data_loaders
    sig = inspect.signature(get_data_loaders)
    print(f'get_data_loaders returns: {sig.return_annotation}')
except Exception as e:
    print(f'ERROR loading mpox_dataset: {e}')
"

###############################################################################
# 9) Training parameters
###############################################################################
SEED=42
MAX_EPOCHS=150
BATCH_SIZE=16
LEARNING_RATE=0.0005  # Reduced from 0.001 for better training
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
    --threshold 0.7 \
    --model_type attention \
    --scheduler cosine \
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
    --threshold 0.7 \
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
