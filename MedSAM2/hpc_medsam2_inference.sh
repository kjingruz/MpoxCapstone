#!/bin/bash
#SBATCH --job-name=MedSAM2_Inference
#SBATCH --output=MedSAM2_Inference_%j.log
#SBATCH --error=MedSAM2_Inference_%j.log
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=main

###############################################################################
# MedSAM2 Inference Script for HPC
# This script runs inference using MedSAM2 on Mpox lesion images.
###############################################################################

# 1) Print start time and node info
echo "=========================================================="
echo "Starting MedSAM2 Inference on HPC"
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
RESULTS_DIR=${BASE_DIR}/results
ENV_SCRIPT=${BASE_DIR}/activate_env.sh

# 3) Activate MedSAM2 environment
echo "Activating MedSAM2 environment..."
source ${ENV_SCRIPT}

# 4) Create results directory
TIMESTAMP=$(date +"%Y%m%d_%H%M")
INFERENCE_DIR=${RESULTS_DIR}/inference_${TIMESTAMP}
mkdir -p ${INFERENCE_DIR}

# 5) Define model checkpoints
SAM2_CHECKPOINT=${CHECKPOINT_DIR}/sam2.1_hiera_base_plus.pt
FINETUNED_MODEL=${BASE_DIR}/medsam2_mpox.pth

# Check if base model exists
if [ ! -f "${SAM2_CHECKPOINT}" ]; then
    echo "ERROR: SAM2 checkpoint not found at ${SAM2_CHECKPOINT}."
    echo "Please download the checkpoint first."
    exit 1
fi

# 6) Check if input data exists
NPZ_DIR=${MPOX_DATA_DIR}/npz_inference

if [ ! -d "${NPZ_DIR}" ] || [ -z "$(ls -A ${NPZ_DIR} 2>/dev/null)" ]; then
    echo "ERROR: No preprocessed inference data found in ${NPZ_DIR}."
    echo "Please run the data preparation script first: sbatch hpc_medsam2_dataprep.sh"
    exit 1
fi

# 7) Run inference with the base model (for comparison)
echo "=========================================================="
echo "Running inference with base SAM2 model (for comparison)..."
echo "=========================================================="

BASE_OUTPUT_DIR=${INFERENCE_DIR}/base_model
mkdir -p ${BASE_OUTPUT_DIR}

python ${SCRIPTS_DIR}/run_medsam2_inference.py \
    --input_dir ${NPZ_DIR} \
    --output_dir ${BASE_OUTPUT_DIR} \
    --sam2_checkpoint ${SAM2_CHECKPOINT} \
    --model_cfg "sam2.1_hiera_b+.yaml" \
    --prompt_method box \
    --bbox_shift 10 \
    --device cuda \
    --num_workers $(nproc)

# 8) Run inference with the fine-tuned model (if available)
if [ -f "${FINETUNED_MODEL}" ]; then
    echo "=========================================================="
    echo "Running inference with fine-tuned MedSAM2 model..."
    echo "=========================================================="
    
    FINETUNED_OUTPUT_DIR=${INFERENCE_DIR}/finetuned_model
    mkdir -p ${FINETUNED_OUTPUT_DIR}
    
    python ${SCRIPTS_DIR}/run_medsam2_inference.py \
        --input_dir ${NPZ_DIR} \
        --output_dir ${FINETUNED_OUTPUT_DIR} \
        --sam2_checkpoint ${SAM2_CHECKPOINT} \
        --medsam2_checkpoint ${FINETUNED_MODEL} \
        --model_cfg "sam2.1_hiera_b+.yaml" \
        --prompt_method box \
        --bbox_shift 10 \
        --device cuda \
        --num_workers $(nproc)
else
    echo "INFO: Fine-tuned model not found at ${FINETUNED_MODEL}."
    echo "Skipping inference with fine-tuned model."
fi

# 9) Create a symbolic link to the latest inference run
LATEST_LINK=${BASE_DIR}/latest_inference
rm -f ${LATEST_LINK} 2>/dev/null
ln -s ${INFERENCE_DIR} ${LATEST_LINK}

# 10) Print summary and next steps
echo "=========================================================="
echo "INFERENCE COMPLETED"
echo "=========================================================="
echo "Inference results: ${INFERENCE_DIR}"
echo "Base model results: ${BASE_OUTPUT_DIR}"
if [ -f "${FINETUNED_MODEL}" ]; then
    echo "Fine-tuned model results: ${FINETUNED_OUTPUT_DIR}"
fi
echo "Symlink to latest run: ${LATEST_LINK}"
echo ""
echo "Next steps:"
echo "1. Evaluate the model performance: sbatch hpc_medsam2_evaluate.sh"
echo "2. Visualize the results in the output directories"
echo "=========================================================="
