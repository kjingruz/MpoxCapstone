#!/bin/bash
#SBATCH --job-name=MedSAM2_DataPrep
#SBATCH --output=MedSAM2_DataPrep_%j.log
#SBATCH --error=MedSAM2_DataPrep_%j.log
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=main

###############################################################################
# MedSAM2 Data Preparation Script for HPC
# This script prepares Mpox images for inference and/or fine-tuning with MedSAM2.
###############################################################################

# 1) Print start time and node info
echo "=========================================================="
echo "Starting MedSAM2 Data Preparation on HPC"
echo "Current time: $(date)"
echo "Running on node: $(hostname)"
echo "Current directory: $(pwd)"
echo "=========================================================="

# 2) Define directories and settings
BASE_DIR=${HOME}/Mpox
MPOX_IMG_DIR=${BASE_DIR}/data/Monkey_Pox
MPOX_MASK_DIR=${MPOX_IMG_DIR}/masks
MPOX_DATA_DIR=${BASE_DIR}/mpox_data
SCRIPTS_DIR=${BASE_DIR}/scripts
ENV_SCRIPT=${BASE_DIR}/activate_env.sh

# 3) Activate MedSAM2 environment
echo "Activating MedSAM2 environment..."
source ${ENV_SCRIPT}

# 4) Create output directories
mkdir -p ${MPOX_DATA_DIR}/{npz_train,npz_val,npz_inference,npy}

# 5) Check for masks directory
if [ ! -d "${MPOX_MASK_DIR}" ] || [ -z "$(ls -A ${MPOX_MASK_DIR} 2>/dev/null)" ]; then
    echo "ERROR: No masks found in ${MPOX_MASK_DIR}."
    echo "Please run the COCO to mask conversion script first: sbatch hpc_medsam2_coco_to_mask.sh"
    exit 1
fi

# 6) Run data preparation for both inference and training
echo "=========================================================="
echo "Preparing Mpox images for inference and training..."
echo "=========================================================="
python ${SCRIPTS_DIR}/mpox_data_prep.py \
    --image_dir ${MPOX_IMG_DIR} \
    --mask_dir ${MPOX_MASK_DIR} \
    --output_dir ${MPOX_DATA_DIR} \
    --mode training \
    --val_ratio 0.2 \
    --target_size 1024 1024 \
    --down_size 256 256 \
    --num_workers $(nproc) \
    --visualize

# Check if data preparation was successful
if [ $? -ne 0 ]; then
    echo "ERROR: Data preparation failed."
    exit 1
fi

# 7) Create a validation sample for visual inspection
echo "Creating validation samples..."
mkdir -p ${BASE_DIR}/validation_samples

# Select a few random samples from training and validation sets
ls ${MPOX_DATA_DIR}/npz_train/*.npz | sort -R | head -n 5 > ${BASE_DIR}/validation_samples/train_samples.txt
ls ${MPOX_DATA_DIR}/npz_val/*.npz | sort -R | head -n 5 > ${BASE_DIR}/validation_samples/val_samples.txt

# Print the samples for manual inspection
echo "Training samples:"
cat ${BASE_DIR}/validation_samples/train_samples.txt
echo "Validation samples:"
cat ${BASE_DIR}/validation_samples/val_samples.txt

# 8) Print summary and next steps
echo "=========================================================="
echo "DATA PREPARATION COMPLETED"
echo "=========================================================="
echo "Prepared data directory: ${MPOX_DATA_DIR}"
echo "Number of training samples: $(ls ${MPOX_DATA_DIR}/npz_train/*.npz | wc -l)"
echo "Number of validation samples: $(ls ${MPOX_DATA_DIR}/npz_val/*.npz | wc -l)"
echo "Number of NpY files for training: $(ls ${MPOX_DATA_DIR}/npy/imgs/*.npy | wc -l)"
echo ""
echo "Next steps:"
echo "1. Fine-tune the model: sbatch hpc_medsam2_finetune.sh"
echo "2. Run inference: sbatch hpc_medsam2_inference.sh"
echo "=========================================================="
