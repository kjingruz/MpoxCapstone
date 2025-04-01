#!/bin/bash

###############################################################################
# Slurm Job Configuration (adjust as needed)
###############################################################################
#SBATCH --job-name=MultiGPU_SimSiam
#SBATCH --output=MultiGPU_SimSiam_%j.log
#SBATCH --error=MultiGPU_SimSiam_%j.log
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
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
VENV_DIR="$HOME/multi_gpu_venv"
SCRIPT_DIR="$(pwd)"
BASE_DIR="$SCRIPT_DIR"
OUTPUT_DIR="$BASE_DIR/pretrained_multi_gpu"
DATA_DIR="$BASE_DIR/data"

# HAM10000 directory
HAM10000_DIR="$DATA_DIR/HAM10000/ISIC-images"  # Path to HAM10000 images

mkdir -p "$OUTPUT_DIR"

###############################################################################
# 3) Check for multiple GPUs
###############################################################################
echo "Checking for GPUs..."
if command -v nvidia-smi &> /dev/null; then
    echo "===== GPU status before job starts ====="
    nvidia-smi
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "========================================="
    echo "Number of GPUs detected: $NUM_GPUS"
    
    if [ "$NUM_GPUS" -lt 2 ]; then
        echo "WARNING: This script is designed for multi-GPU training, but only $NUM_GPUS GPU was detected."
        echo "The script will still run, but won't use distributed training."
    fi
else
    echo "ERROR: No GPUs detected. This script requires at least one GPU."
    exit 1
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

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA 11.8 support"
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

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
# 7) Copy multi-GPU pretraining script if needed
###############################################################################
if [ ! -f "$SCRIPT_DIR/pretrain_ham10000_multi_gpu.py" ]; then
    echo "ERROR: Multi-GPU pretraining script not found."
    echo "Please make sure pretrain_ham10000_multi_gpu.py is in the current directory."
    exit 1
fi

###############################################################################
# 8) Training parameters
###############################################################################
BATCH_SIZE=64  # Per GPU batch size (increased for multi-GPU)
EPOCHS=50
LEARNING_RATE=0.05
IMAGE_SIZE=256
NUM_WORKERS=4  # Workers per GPU
WORLD_SIZE=$NUM_GPUS  # Use all available GPUs

###############################################################################
# 9) Run multi-GPU pretraining
###############################################################################
echo "Starting multi-GPU self-supervised pretraining on HAM10000 with $NUM_GPUS GPUs..."

# Set NCCL environment variables for better performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=^docker0,lo

# Run distributed training script
python pretrain_ham10000_multi_gpu.py \
    --ham10000_dir "$HAM10000_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --img_size $IMAGE_SIZE \
    --model_type attention \
    --num_workers $NUM_WORKERS \
    --world_size $WORLD_SIZE \
    2>&1 | tee multi_gpu_pretrain_log.txt

# Check if pretraining was successful
if [ $? -ne 0 ]; then
    echo "ERROR: Multi-GPU pretraining failed. See multi_gpu_pretrain_log.txt for details."
    exit 1
fi

# Find the final pretrained encoder
PRETRAINED_MODEL=$(find "$OUTPUT_DIR" -name "encoder_final.pth" | sort -t_ -k2 -n | tail -1)

if [ -z "$PRETRAINED_MODEL" ]; then
    echo "WARNING: Final encoder not found. Looking for any checkpoint..."
    PRETRAINED_MODEL=$(find "$OUTPUT_DIR" -name "simsiam_ep*.pth" | sort -t_ -k2 -n | tail -1)

    if [ -z "$PRETRAINED_MODEL" ]; then
        echo "ERROR: No pretrained model found. Pretraining may have failed."
        exit 1
    fi
fi

echo "Multi-GPU pretraining completed successfully!"
echo "Pretrained model saved to: $PRETRAINED_MODEL"

###############################################################################
# 10) Final status and summary
###############################################################################
echo "============================================================="
echo "MULTI-GPU PRETRAINING COMPLETED SUCCESSFULLY"
echo "============================================================="
echo "Trained with $NUM_GPUS GPUs"
echo "Pretrained model: $PRETRAINED_MODEL"
echo "Output directory: $OUTPUT_DIR"
echo "============================================================="

# Display GPU status after job completion
if command -v nvidia-smi &> /dev/null; then
    echo "===== GPU status after job completion ====="
    nvidia-smi
    echo "=========================================="
fi
