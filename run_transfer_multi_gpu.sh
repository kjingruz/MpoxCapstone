#!/bin/bash

###############################################################################
# Slurm Job Configuration (adjust as needed)
###############################################################################
#SBATCH --job-name=MultiGPU_Transfer
#SBATCH --output=MultiGPU_Transfer_%j.log
#SBATCH --error=MultiGPU_Transfer_%j.log
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
OUTPUT_DIR="$BASE_DIR/transfer_multi_gpu_outputs"
DATA_DIR="$BASE_DIR/data"

# Specific dataset directories
PH2_DIR="$DATA_DIR/PH2_processed"
MPOX_DIR="$DATA_DIR/Monkey_Pox"

# Path to the pretrained encoder from the previous step
PRETRAINED_MODEL="/home/zhangk/Mpox/pretrained_multi_gpu/simsiam_run_20250331_225051/encoder_final.pth"

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
echo "Setting up virtual environment..."

# Check if virtual environment exists, create if not
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
else
    echo "Using existing virtual environment at $VENV_DIR"
fi

# Activate the virtual environment
echo "Activating virtual environment"
source "$VENV_DIR/bin/activate"

# Verify activation worked
python_path=$(which python)
echo "Using Python from: $python_path"

# Upgrade pip and install required packages
echo "Installing required packages..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo "Installing other dependencies..."
pip install numpy pandas matplotlib scikit-learn tqdm
pip install Pillow opencv-python-headless==4.7.0.72 albumentations

# Verify installations
echo "Verifying installations..."
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
python -c "import albumentations; print('Albumentations version:', albumentations.__version__)"

###############################################################################
# 5) Make sure the UNetEncoder class is available
###############################################################################
if [ ! -f "$SCRIPT_DIR/unet_encoder.py" ]; then
    echo "Creating unet_encoder.py file..."
    cat > "$SCRIPT_DIR/unet_encoder.py" << 'EOF'
import torch
import torch.nn as nn

class UNetEncoder(nn.Module):
    """
    Extract encoder part from UNet for self-supervised learning and transfer learning
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
EOF
    echo "Created unet_encoder.py"
fi

###############################################################################
# 6) Copy the multi-GPU transfer learning script if it doesn't exist
###############################################################################
if [ ! -f "$SCRIPT_DIR/train_transfer_multi_gpu.py" ]; then
    echo "ERROR: train_transfer_multi_gpu.py not found in $SCRIPT_DIR"
    echo "Please create this file first."
    exit 1
fi

###############################################################################
# 7) Configure matplotlib for non-interactive backend
###############################################################################
mkdir -p ~/.config/matplotlib
echo "backend: Agg" > ~/.config/matplotlib/matplotlibrc

###############################################################################
# 8) Training parameters
###############################################################################
BATCH_SIZE=16  # Per GPU
EPOCHS=100
LEARNING_RATE=0.001
IMAGE_SIZE=256
NUM_WORKERS=4  # Per GPU
FINE_TUNE_EPOCH=50  # When to unfreeze encoder
WORLD_SIZE=$NUM_GPUS  # Use all available GPUs
ACCUMULATION_STEPS=2  # Effectively doubles the batch size

###############################################################################
# 9) Set environment variables for better distributed performance
###############################################################################
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^docker0,lo

# Set PyTorch environment variables for debugging
export TORCH_DISTRIBUTED_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1

###############################################################################
# 10) Run multi-GPU transfer learning
###############################################################################
echo "Starting multi-GPU transfer learning with $NUM_GPUS GPUs..."

# Run the transfer learning script with pretrained weights
python train_transfer_multi_gpu.py \
    --ph2_dir "$PH2_DIR" \
    --mpox_dir "$MPOX_DIR" \
    --pretrained_weights "$PRETRAINED_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --img_size $IMAGE_SIZE \
    --fine_tune_epoch $FINE_TUNE_EPOCH \
    --model_type attention \
    --scheduler cosine \
    --num_workers $NUM_WORKERS \
    --world_size $WORLD_SIZE \
    --accumulation_steps $ACCUMULATION_STEPS \
    --freeze_encoder \
    --save_interval 10 \
    --early_stopping_patience 15 \
    2>&1 | tee multi_gpu_transfer_log.txt

# Check if transfer learning was successful
if [ $? -ne 0 ]; then
    echo "ERROR: Multi-GPU transfer learning failed. See multi_gpu_transfer_log.txt for details."
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
# 11) Final status and summary
###############################################################################
echo "============================================================="
echo "MULTI-GPU TRANSFER LEARNING COMPLETED SUCCESSFULLY"
echo "============================================================="
echo "Used pretrained model: $PRETRAINED_MODEL"
echo "Trained on PH2 dataset with ${NUM_GPUS} GPUs"
echo "Tested on Mpox dataset"
echo "Best model saved to: $BEST_MODEL"
echo "============================================================="
echo "Output directory: $OUTPUT_DIR"
echo "============================================================="

# Display GPU status after job completion
if command -v nvidia-smi &> /dev/null; then
    echo "===== GPU status after job completion ====="
    nvidia-smi
    echo "=========================================="
fi
