#!/bin/bash

###############################################################################
# Slurm Job Configuration
###############################################################################
#SBATCH --job-name=CrossDataset_UNet
#SBATCH --output=CrossDataset_UNet_%j.log
#SBATCH --error=CrossDataset_UNet_%j.log
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=main

###############################################################################
# 1) Load modules 
###############################################################################
echo "Loading HPC modules..."
module load python/3.10
module load cuda/11.8.0
module load cudnn/8.6.0

###############################################################################
# 2) Setup directories
###############################################################################
VENV_DIR="$HOME/cross_dataset_venv"
SCRIPT_DIR="$(pwd)"
BASE_DIR="$SCRIPT_DIR"
OUTPUT_DIR="$BASE_DIR/cross_dataset_outputs"
DATA_DIR="$BASE_DIR/data"

# Specific dataset directories
PH2_DIR="$DATA_DIR/PH2_processed"
MPOX_DIR="$DATA_DIR/Monkey_Pox"

mkdir -p "$OUTPUT_DIR"

###############################################################################
# 3) Check for GPU
###############################################################################
echo "Checking for GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "===== GPU status before job starts ====="
    nvidia-smi
    echo "========================================="
    GPU_FLAG=""
else
    echo "WARNING: No GPU detected, using CPU only. This is not recommended for training."
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
# 5) Install Required Libraries
###############################################################################
echo "Installing required Python packages..."

# Install NumPy 1.x first
pip install "numpy<2.0.0"

# Install PyTorch with CUDA support if GPU is available
if [ -z "$GPU_FLAG" ]; then
    echo "Installing PyTorch with CUDA 11.8 support"
    pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
else
    echo "Installing PyTorch for CPU"
    pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
fi

# Install other required packages
pip install pandas matplotlib scikit-learn tqdm
pip install Pillow
pip install opencv-python-headless==4.7.0.72
pip install albumentations

# Configure matplotlib for non-interactive backend
mkdir -p ~/.config/matplotlib
echo "backend: Agg" > ~/.config/matplotlib/matplotlibrc

###############################################################################
# 6) Copy updated scripts
###############################################################################
echo "Copying updated scripts for cross-dataset training..."

# Copy necessary scripts if they're not already in the working directory
# Assuming you've already created and downloaded these files from previous steps
if [ ! -f "$SCRIPT_DIR/cross_dataset_loader.py" ]; then
    echo "Error: cross_dataset_loader.py not found. Please create this file first."
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/self_supervised.py" ]; then
    echo "Error: self_supervised.py not found. Please create this file first."
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/train_cross_dataset.py" ]; then
    echo "Error: train_cross_dataset.py not found. Please create this file first."
    exit 1
fi

###############################################################################
# 7) Verify dataset structure
###############################################################################
echo "Verifying dataset structure..."

# Check PH2 dataset
if [ ! -d "$PH2_DIR/split/train/images" ] || [ ! -d "$PH2_DIR/split/train/masks" ]; then
    echo "Error: PH2 dataset structure is not as expected. Please check the paths."
    echo "Expected: $PH2_DIR/split/train/images and $PH2_DIR/split/train/masks"
    exit 1
fi

# Check Mpox dataset
if [ ! -d "$MPOX_DIR" ]; then
    echo "Error: Mpox dataset directory not found."
    echo "Expected: $MPOX_DIR"
    exit 1
fi

echo "Dataset structure verified."

###############################################################################
# 8) Training parameters
###############################################################################
SEED=42
MAX_EPOCHS=150
BATCH_SIZE=16
LEARNING_RATE=0.0005
IMAGE_SIZE=256
NUM_WORKERS=4
USE_PRETRAINED=true  # Set to true to enable self-supervised pretraining
PRETRAIN_METHOD="simsiam"
PRETRAIN_EPOCHS=30

###############################################################################
# 9) Verify CUDA is available in Python
###############################################################################
echo "Verifying CUDA is available to PyTorch..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')" || {
    echo "WARNING: Unable to verify CUDA availability. Check PyTorch installation."
}

###############################################################################
# 10) Run cross-dataset training
###############################################################################
echo "Starting cross-dataset U-Net training..."

# Build the command with all parameters
TRAIN_CMD="python train_cross_dataset.py \
    --ph2_dir $PH2_DIR \
    --mpox_dir $MPOX_DIR \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --epochs $MAX_EPOCHS \
    --lr $LEARNING_RATE \
    --img_size $IMAGE_SIZE \
    --num_workers $NUM_WORKERS \
    --threshold 0.7 \
    --model_type attention \
    --scheduler cosine"

# Add pretraining if enabled
if [ "$USE_PRETRAINED" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --use_pretrained --pretrain_method $PRETRAIN_METHOD --pretrain_epochs $PRETRAIN_EPOCHS"
fi

# Add GPU flag if needed
TRAIN_CMD="$TRAIN_CMD $GPU_FLAG"

# Run the command and log output
echo "Running: $TRAIN_CMD"
eval "$TRAIN_CMD 2>&1 | tee cross_dataset_training_log.txt"

# Check if training was successful
if [ $? -ne 0 ]; then
    echo "ERROR: Training failed. See cross_dataset_training_log.txt for details."
    exit 1
fi

###############################################################################
# 11) Find best model
###############################################################################
BEST_MODEL=$(find "$OUTPUT_DIR" -name "best_model.pth" | sort -t_ -k2 -n | tail -1)

if [ -z "$BEST_MODEL" ]; then
    echo "WARNING: No best model found! Checking for any checkpoint..."
    BEST_MODEL=$(find "$OUTPUT_DIR" -name "*.pth" | head -1)

    if [ -z "$BEST_MODEL" ]; then
        echo "ERROR: No model checkpoints found. Training may have failed."
        exit 1
    fi
fi

echo "Best model found: $BEST_MODEL"

###############################################################################
# 12) Run inference on Mpox dataset
###############################################################################
echo "Running inference on Mpox data using cross-dataset model..."

python inference.py \
    --model "$BEST_MODEL" \
    --input "$MPOX_DIR" \
    --output "$OUTPUT_DIR/mpox_results" \
    --img_size "$IMAGE_SIZE" \
    --threshold 0.7 \
    $GPU_FLAG 2>&1 | tee cross_dataset_inference_log.txt

# Check if inference was successful
if [ $? -ne 0 ]; then
    echo "ERROR: Inference failed. See cross_dataset_inference_log.txt for details."
    exit 1
fi

###############################################################################
# 13) Final status and summary
###############################################################################
echo "============================================================="
echo "CROSS-DATASET TRAINING COMPLETED SUCCESSFULLY"
echo "============================================================="
echo "Training approach: Train on PH2, Test on Mpox"
if [ "$USE_PRETRAINED" = true ]; then
    echo "Used self-supervised pretraining with method: $PRETRAIN_METHOD"
fi
echo "Results saved to: $OUTPUT_DIR"
echo "Best model saved to: $BEST_MODEL"
echo "============================================================="

# Display GPU status after job completion
if command -v nvidia-smi &> /dev/null; then
    echo "===== GPU status after job completion ====="
    nvidia-smi
    echo "=========================================="
fi
