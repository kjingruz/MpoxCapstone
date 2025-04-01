#!/bin/bash

###############################################################################
# Slurm Job Configuration
###############################################################################
#SBATCH --job-name=TransferLearn_UNet
#SBATCH --output=TransferLearn_UNet_%j.log
#SBATCH --error=TransferLearn_UNet_%j.log
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=700G
#SBATCH --gres=gpu:4
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
VENV_DIR="$HOME/transfer_learning_venv"
SCRIPT_DIR="$(pwd)"
BASE_DIR="$SCRIPT_DIR"
OUTPUT_DIR="$BASE_DIR/transfer_learning_outputs"
PRETRAIN_DIR="$BASE_DIR/pretrained"
DATA_DIR="$BASE_DIR/data"

# Specific dataset directories
PH2_DIR="$DATA_DIR/PH2_processed"
MPOX_DIR="$DATA_DIR/Monkey_Pox"
HAM10000_DIR="$DATA_DIR/HAM10000/ISIC-images"  # Path to HAM10000 images

mkdir -p "$OUTPUT_DIR"
mkdir -p "$PRETRAIN_DIR"

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
# 6) Check if HAM10000 dataset exists
###############################################################################
if [ ! -d "$HAM10000_DIR" ]; then
    echo "ERROR: HAM10000 dataset directory not found: $HAM10000_DIR"
    echo "Please download and extract the dataset first."
    exit 1
fi

# Quick check to confirm files exist
HAM_FILE_COUNT=$(ls "$HAM10000_DIR" | grep -c "ISIC_")
if [ "$HAM_FILE_COUNT" -eq 0 ]; then
    echo "ERROR: No ISIC image files found in $HAM10000_DIR"
    exit 1
fi

echo "Found approximately $HAM_FILE_COUNT HAM10000 image files."

###############################################################################
# 7) Training parameters
###############################################################################
BATCH_SIZE=32
PRETRAIN_EPOCHS=50
FINETUNE_EPOCHS=100
LEARNING_RATE_PRETRAIN=0.05  # Higher LR for pretraining
LEARNING_RATE_FINETUNE=0.001  # Lower LR for fine-tuning
IMAGE_SIZE=256
NUM_WORKERS=4
FINETUNE_EPOCH=50  # When to unfreeze encoder
FREEZE_ENCODER=true

###############################################################################
# 8) Step 1: Self-supervised pretraining on HAM10000
###############################################################################
echo "Step 1: Starting self-supervised pretraining on HAM10000..."

python pretrain_ham10000.py \
    --ham10000_dir "$HAM10000_DIR" \
    --output_dir "$PRETRAIN_DIR" \
    --batch_size $BATCH_SIZE \
    --epochs $PRETRAIN_EPOCHS \
    --lr $LEARNING_RATE_PRETRAIN \
    --img_size $IMAGE_SIZE \
    --model_type attention \
    --num_workers $NUM_WORKERS \
    $GPU_FLAG 2>&1 | tee pretrain_log.txt

# Check if pretraining was successful
if [ $? -ne 0 ]; then
    echo "ERROR: Pretraining failed. See pretrain_log.txt for details."
    exit 1
fi

# Find the final pretrained encoder
PRETRAINED_MODEL=$(find "$PRETRAIN_DIR" -name "encoder_final.pth" | sort -t_ -k2 -n | tail -1)

if [ -z "$PRETRAINED_MODEL" ]; then
    echo "WARNING: Final encoder not found. Looking for any checkpoint..."
    PRETRAINED_MODEL=$(find "$PRETRAIN_DIR" -name "simsiam_ep*.pth" | sort -t_ -k2 -n | tail -1)

    if [ -z "$PRETRAINED_MODEL" ]; then
        echo "ERROR: No pretrained model found. Pretraining may have failed."
        exit 1
    fi
fi

echo "Using pretrained model: $PRETRAINED_MODEL"

###############################################################################
# 9) Step 2: Transfer learning from HAM10000 to PH2
###############################################################################
echo "Step 2: Starting transfer learning from HAM10000 to PH2..."

# Make sure the necessary helper scripts exist
for script in cross_dataset_loader.py ham10000_loader.py train_transfer.py; do
    if [ ! -f "$SCRIPT_DIR/$script" ]; then
        echo "ERROR: Required script $script not found in $SCRIPT_DIR"
        exit 1
    fi
done

echo "All required scripts found."

FREEZE_FLAG=""
if [ "$FREEZE_ENCODER" = true ]; then
    FREEZE_FLAG="--freeze_encoder"
fi

python train_transfer.py \
    --ph2_dir "$PH2_DIR" \
    --mpox_dir "$MPOX_DIR" \
    --pretrained_weights "$PRETRAINED_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --epochs $FINETUNE_EPOCHS \
    --lr $LEARNING_RATE_FINETUNE \
    --img_size $IMAGE_SIZE \
    --fine_tune_epoch $FINETUNE_EPOCH \
    --model_type attention \
    --scheduler cosine \
    --num_workers $NUM_WORKERS \
    $FREEZE_FLAG \
    $GPU_FLAG 2>&1 | tee transfer_learning_log.txt

# Check if transfer learning was successful
if [ $? -ne 0 ]; then
    echo "ERROR: Transfer learning failed. See transfer_learning_log.txt for details."
    exit 1
fi

# Find the best model
BEST_MODEL=$(find "$OUTPUT_DIR" -name "best_model.pth" | head -1)

if [ -z "$BEST_MODEL" ]; then
    echo "WARNING: No best model found. Looking for any checkpoint..."
    BEST_MODEL=$(find "$OUTPUT_DIR" -name "model_epoch_*.pth" | sort -t_ -k3 -n | tail -1)

    if [ -z "$BEST_MODEL" ]; then
        echo "ERROR: No model checkpoints found. Training may have failed."
        exit 1
    fi
fi

echo "Best model: $BEST_MODEL"

###############################################################################
# 10) Final status and summary
###############################################################################
echo "============================================================="
echo "COMPLETE PIPELINE EXECUTED SUCCESSFULLY"
echo "============================================================="
echo "Step 1: Self-supervised pretraining on HAM10000 - COMPLETED"
echo "       Pretrained model: $PRETRAINED_MODEL"
echo ""
echo "Step 2: Transfer learning from HAM10000 to PH2 - COMPLETED"
echo "       Best model: $BEST_MODEL"
echo "============================================================="
echo "Output directories:"
echo "  - Pretraining: $PRETRAIN_DIR"
echo "  - Transfer Learning: $OUTPUT_DIR"
echo "============================================================="

# Display GPU status after job completion
if command -v nvidia-smi &> /dev/null; then
    echo "===== GPU status after job completion ====="
    nvidia-smi
    echo "=========================================="
fi
