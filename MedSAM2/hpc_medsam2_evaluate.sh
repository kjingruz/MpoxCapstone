#!/bin/bash
#SBATCH --job-name=MedSAM2_Evaluate
#SBATCH --output=MedSAM2_Evaluate_%j.log
#SBATCH --error=MedSAM2_Evaluate_%j.log
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=main

###############################################################################
# MedSAM2 Evaluation Script for HPC
# This script evaluates and compares segmentation performance of different models.
###############################################################################

# 1) Print start time and node info
echo "=========================================================="
echo "Starting MedSAM2 Evaluation on HPC"
echo "Current time: $(date)"
echo "Running on node: $(hostname)"
echo "Current directory: $(pwd)"
echo "=========================================================="

# 2) Define directories and settings
BASE_DIR=${HOME}/Mpox
MPOX_DATA_DIR=${BASE_DIR}/mpox_data
SCRIPTS_DIR=${BASE_DIR}/scripts
RESULTS_DIR=${BASE_DIR}/results
EVALUATION_DIR=${BASE_DIR}/evaluation
ENV_SCRIPT=${BASE_DIR}/activate_env.sh

# 3) Activate MedSAM2 environment
echo "Activating MedSAM2 environment..."
source ${ENV_SCRIPT}

# 4) Create evaluation directory
TIMESTAMP=$(date +"%Y%m%d_%H%M")
EVAL_OUTPUT_DIR=${EVALUATION_DIR}/${TIMESTAMP}
mkdir -p ${EVAL_OUTPUT_DIR}

# 5) Find the latest inference results
LATEST_INFERENCE=${BASE_DIR}/latest_inference

if [ ! -d "${LATEST_INFERENCE}" ]; then
    echo "ERROR: Latest inference directory not found at ${LATEST_INFERENCE}."
    echo "Please run inference first: sbatch hpc_medsam2_inference.sh"
    exit 1
fi

# 6) Locate the prediction and ground truth directories
BASE_PRED_DIR=${LATEST_INFERENCE}/base_model/masks
FINETUNED_PRED_DIR=${LATEST_INFERENCE}/finetuned_model/masks
GT_DIR=${MPOX_DATA_DIR}/npz_val  # Ground truth masks are in the validation set

# Check if prediction directories exist
if [ ! -d "${BASE_PRED_DIR}" ]; then
    echo "ERROR: Base model predictions not found at ${BASE_PRED_DIR}."
    exit 1
fi

# 7) Run evaluation for base model
echo "=========================================================="
echo "Evaluating base SAM2 model..."
echo "=========================================================="

BASE_EVAL_DIR=${EVAL_OUTPUT_DIR}/base_model
mkdir -p ${BASE_EVAL_DIR}

python ${SCRIPTS_DIR}/evaluate_medsam2.py \
    --pred_dir ${BASE_PRED_DIR} \
    --gt_dir ${GT_DIR} \
    --output_dir ${BASE_EVAL_DIR} \
    --num_workers $(nproc)

# 8) Run evaluation for fine-tuned model (if available)
if [ -d "${FINETUNED_PRED_DIR}" ]; then
    echo "=========================================================="
    echo "Evaluating fine-tuned MedSAM2 model..."
    echo "=========================================================="
    
    FINETUNED_EVAL_DIR=${EVAL_OUTPUT_DIR}/finetuned_model
    mkdir -p ${FINETUNED_EVAL_DIR}
    
    python ${SCRIPTS_DIR}/evaluate_medsam2.py \
        --pred_dir ${FINETUNED_PRED_DIR} \
        --gt_dir ${GT_DIR} \
        --output_dir ${FINETUNED_EVAL_DIR} \
        --num_workers $(nproc)
    
    # 9) Compare the two models
    echo "=========================================================="
    echo "Comparing base and fine-tuned models..."
    echo "=========================================================="
    
    COMPARISON_DIR=${EVAL_OUTPUT_DIR}/comparison
    mkdir -p ${COMPARISON_DIR}
    
    python ${SCRIPTS_DIR}/evaluate_medsam2.py \
        --pred_dir ${BASE_PRED_DIR} \
        --gt_dir ${GT_DIR} \
        --output_dir ${COMPARISON_DIR} \
        --compare ${FINETUNED_PRED_DIR} \
        --model_names "SAM2-Base" "MedSAM2-Mpox" \
        --num_workers $(nproc)
else
    echo "INFO: Fine-tuned model predictions not found at ${FINETUNED_PRED_DIR}."
    echo "Skipping fine-tuned model evaluation and comparison."
fi

# 10) Create a symbolic link to the latest evaluation
LATEST_EVAL_LINK=${BASE_DIR}/latest_evaluation
rm -f ${LATEST_EVAL_LINK} 2>/dev/null
ln -s ${EVAL_OUTPUT_DIR} ${LATEST_EVAL_LINK}

# 11) Print summary and next steps
echo "=========================================================="
echo "EVALUATION COMPLETED"
echo "=========================================================="
echo "Evaluation results: ${EVAL_OUTPUT_DIR}"
echo "Base model evaluation: ${BASE_EVAL_DIR}"
if [ -d "${FINETUNED_PRED_DIR}" ]; then
    echo "Fine-tuned model evaluation: ${FINETUNED_EVAL_DIR}"
    echo "Model comparison: ${COMPARISON_DIR}"
fi
echo "Symlink to latest evaluation: ${LATEST_EVAL_LINK}"
echo ""
echo "Next steps:"
echo "1. Review the evaluation metrics and visualizations"
echo "2. Check the model comparison results for improvements"
echo "=========================================================="
