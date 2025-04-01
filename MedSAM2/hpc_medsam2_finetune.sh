#!/bin/bash
#SBATCH --job-name=MedSAM2_Finetune
#SBATCH --output=MedSAM2_Finetune_%j.log
#SBATCH --error=MedSAM2_Finetune_%j.log
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=700G
#SBATCH --gres=gpu:4
#SBATCH --partition=main

###############################################################################
# MedSAM2 Fine-tuning Script for HPC
# This script fine-tunes SAM2 base_plus model on Mpox lesion data
###############################################################################

# 1) Print start time and node info
echo "=========================================================="
echo "Starting MedSAM2 Fine-tuning on HPC"
echo "Current time: $(date)"
echo "Running on node: $(hostname)"
echo "GPU information:"
nvidia-smi
echo "Current directory: $(pwd)"
echo "=========================================================="

# 2) Define directories and settings
BASE_DIR=${HOME}/Mpox
MPOX_DATA_DIR=${BASE_DIR}/mpox_data
SCRIPTS_DIR=${BASE_DIR}/scripts
CHECKPOINT_DIR=${BASE_DIR}/checkpoints
FINETUNE_DIR=${BASE_DIR}/finetune
ENV_SCRIPT=${BASE_DIR}/activate_env.sh

# 3) Activate MedSAM2 environment
echo "Activating MedSAM2 environment..."
source ${ENV_SCRIPT}

# 4) Create fine-tuning output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M")
FINETUNE_OUTPUT_DIR=${FINETUNE_DIR}/${TIMESTAMP}
mkdir -p ${FINETUNE_OUTPUT_DIR}

# 5) Check for preprocessed training data
NPY_DIR=${MPOX_DATA_DIR}/npy

if [ ! -d "${NPY_DIR}" ] || [ -z "$(ls -A ${NPY_DIR}/imgs 2>/dev/null)" ]; then
    echo "ERROR: No preprocessed training data found in ${NPY_DIR}."
    echo "Please run the data preparation script first: sbatch hpc_medsam2_dataprep.sh"
    exit 1
fi

# 6) Set SAM2 checkpoint path
SAM2_CHECKPOINT=${CHECKPOINT_DIR}/sam2.1_hiera_base_plus.pt

if [ ! -f "${SAM2_CHECKPOINT}" ]; then
    echo "ERROR: SAM2 checkpoint not found at ${SAM2_CHECKPOINT}."
    echo "Please download the checkpoint first."
    exit 1
fi

# 7) Set batch size based on available GPU memory
# With 4 V100 GPUs (32GB each), we can use a larger batch size
BATCH_SIZE=32

# Adjust batch size if needed based on model size
if [[ $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1) -lt 16000 ]]; then
    # Less than 16GB GPU memory available, reduce batch size
    BATCH_SIZE=16
    echo "Limited GPU memory detected, reducing batch size to ${BATCH_SIZE}"
fi

# 8) Run fine-tuning
echo "=========================================================="
echo "Running MedSAM2 fine-tuning on Mpox data..."
echo "Checkpoint: ${SAM2_CHECKPOINT}"
echo "Data directory: ${NPY_DIR}"
echo "Batch size: ${BATCH_SIZE}"
echo "Output directory: ${FINETUNE_OUTPUT_DIR}"
echo "=========================================================="

python ${SCRIPTS_DIR}/finetune_medsam2_mpox.py \
    --data_dir ${NPY_DIR} \
    --output_dir ${FINETUNE_OUTPUT_DIR} \
    --sam2_checkpoint ${SAM2_CHECKPOINT} \
    --model_cfg "sam2.1_hiera_b+.yaml" \
    --batch_size ${BATCH_SIZE} \
    --num_epochs 30 \
    --learning_rate 1e-5 \
    --bbox_shift 10 \
    --device cuda \
    --num_workers 16 \
    --vis_samples 8

# 9) Check if fine-tuning was successful
if [ $? -ne 0 ]; then
    echo "ERROR: Fine-tuning failed. Check logs for details."
    exit 1
fi

# 10) Create a symbolic link to the latest fine-tuning run
LATEST_LINK=${BASE_DIR}/latest_finetune
rm -f ${LATEST_LINK} 2>/dev/null
ln -s ${FINETUNE_OUTPUT_DIR} ${LATEST_LINK}

# 11) Copy the final model to a standard location
FINAL_MODEL=${FINETUNE_OUTPUT_DIR}/medsam2_mpox_final.pth
STANDARD_MODEL=${BASE_DIR}/medsam2_mpox.pth
cp ${FINAL_MODEL} ${STANDARD_MODEL}

# 12) Print summary and next steps
echo "=========================================================="
echo "FINE-TUNING COMPLETED"
echo "=========================================================="
echo "Fine-tuning results: ${FINETUNE_OUTPUT_DIR}"
echo "Final model: ${STANDARD_MODEL}"
echo "Symlink to latest run: ${LATEST_LINK}"
echo ""
echo "Next steps:"
echo "1. Run inference with the fine-tuned model: sbatch hpc_medsam2_inference.sh"
echo "2. Evaluate the model performance: sbatch hpc_medsam2_evaluate.sh"
echo "=========================================================="
