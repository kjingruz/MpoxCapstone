#!/bin/bash
# Training pipeline for Mask R-CNN on HPC
#SBATCH --job-name=train_maskrcnn
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=train_maskrcnn.%j.out

# Load modules (adjust based on your HPC configuration)
module load anaconda3
module load cuda/11.2
module load cudnn/8.1.0

# Activate the environment
source activate lesion_detection

# Set paths
PROJECT_DIR=~/lesion_detection_project
DATA_DIR=$PROJECT_DIR/data
MODELS_DIR=$PROJECT_DIR/models
CODE_DIR=$PROJECT_DIR/code

# Step 1: Preprocess the datasets
echo "Step 1: Preprocessing datasets..."
python $CODE_DIR/preprocess_datasets.py \
    --output_dir $DATA_DIR/processed \
    --ph2_dir $DATA_DIR/raw/PH2_Dataset_images \
    --coco_json $DATA_DIR/raw/mpox_annotations.json \
    --coco_images_dir $DATA_DIR/raw/mpox_images \
    --visualize

# Step 2: Train the Mask R-CNN model using the Matterport implementation
echo "Step 2: Training Mask R-CNN model..."
python $CODE_DIR/train_maskrcnn.py \
    --dataset $DATA_DIR/processed \
    --model_dir $MODELS_DIR \
    --weights coco \
    --epochs 50 \
    --steps_per_epoch 500 \
    --batch_size 2

echo "Training complete. Model saved to $MODELS_DIR."
