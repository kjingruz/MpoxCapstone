#!/bin/bash
# SLURM script for training the Mask R-CNN model on PH2 + Mpox datasets
#SBATCH --job-name=lesion_detection
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=lesion_detection_%j.out

# Load modules (adjust based on your HPC configuration)
module load anaconda3
module load cuda/11.2
module load cudnn/8.1.0

# Activate the environment
source activate lesion_detection

# Set directories
PROJECT_DIR=~/lesion_detection_project
MASK_RCNN_DIR=$PROJECT_DIR/Mask_RCNN
PH2_DIR=$PROJECT_DIR/data/PH2_processed
MPOX_DIR=$PROJECT_DIR/data/Monkey_Pox
OUTPUT_DIR=$PROJECT_DIR/results/$(date +"%Y%m%d_%H%M%S")

# Create output directory
mkdir -p $OUTPUT_DIR

# Clone Mask R-CNN repository if it doesn't exist
if [ ! -d "$MASK_RCNN_DIR" ]; then
    echo "Cloning Mask R-CNN repository..."
    git clone https://github.com/matterport/Mask_RCNN.git $MASK_RCNN_DIR
fi

# Download COCO weights if they don't exist
COCO_WEIGHTS=$MASK_RCNN_DIR/mask_rcnn_coco.h5
if [ ! -f "$COCO_WEIGHTS" ]; then
    echo "Downloading COCO weights..."
    wget -O $COCO_WEIGHTS https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
fi

# Copy the dataset loader and training scripts to the project directory
cp data_loader.py $PROJECT_DIR/
cp train.py $PROJECT_DIR/

# Change to project directory
cd $PROJECT_DIR

# Print system information
echo "CUDA version: $(nvcc --version | grep release | awk '{print $6}')"
echo "Python version: $(python --version)"
echo "TensorFlow version: $(python -c 'import tensorflow as tf; print(tf.__version__)')"
echo "GPU information:"
nvidia-smi

# Run the training script
echo "Starting training at $(date)"
python train.py \
    --ph2_dir $PH2_DIR \
    --mpox_dir $MPOX_DIR \
    --output_dir $OUTPUT_DIR \
    --weights coco \
    --batch_size 2 \
    --steps_per_epoch 200 \
    --heads_epochs 20 \
    --all_epochs 40 \
    --visualize \
    --test_inference

echo "Training completed at $(date)"

# Archive the results
ARCHIVE_NAME="lesion_detection_$(date +"%Y%m%d_%H%M%S").tar.gz"
echo "Archiving results to $ARCHIVE_NAME"
tar -czf $PROJECT_DIR/$ARCHIVE_NAME -C $OUTPUT_DIR .

echo "Done!"
