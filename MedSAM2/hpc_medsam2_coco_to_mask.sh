#!/bin/bash
#SBATCH --job-name=Mpox_COCO2Mask
#SBATCH --output=Mpox_COCO2Mask_%j.log
#SBATCH --error=Mpox_COCO2Mask_%j.log
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=main

###############################################################################
# MedSAM2 COCO to Mask Conversion Script for HPC
# This script converts COCO format annotations to binary masks for Mpox lesions
###############################################################################

# 1) Print start time and node info
echo "=========================================================="
echo "Starting COCO to Mask Conversion for Mpox Dataset"
echo "Current time: $(date)"
echo "Running on node: $(hostname)"
echo "Current directory: $(pwd)"
echo "=========================================================="

# 2) Define directories and settings
BASE_DIR=${HOME}/Mpox
MPOX_DIR=${BASE_DIR}/data/Monkey_Pox
COCO_FILE=${MPOX_DIR}/annotation/instances_default.json
SCRIPTS_DIR=${BASE_DIR}/scripts
MASK_DIR=${MPOX_DIR}/masks
ENV_SCRIPT=${BASE_DIR}/activate_env.sh

# 3) Activate MedSAM2 environment if available
if [ -f "${ENV_SCRIPT}" ]; then
    echo "Activating MedSAM2 environment..."
    source ${ENV_SCRIPT}
else
    echo "No environment activation script found. Using system Python."
fi

# 4) Create output directory for masks
mkdir -p ${MASK_DIR}

# 5) Run COCO to mask conversion
echo "Converting COCO annotations to binary masks..."
python ${SCRIPTS_DIR}/coco_to_masks_simple.py \
    --coco_json ${COCO_FILE} \
    --img_dir ${MPOX_DIR} \
    --output_dir ${MASK_DIR} \
    --visualize

# 6) Print summary
echo "=========================================================="
echo "COCO TO MASK CONVERSION COMPLETED"
echo "=========================================================="
echo "Input COCO file: ${COCO_FILE}"
echo "Output mask directory: ${MASK_DIR}"
echo "Check the masks directory to verify the conversion."
echo "Next step: Run data preparation script: sbatch hpc_medsam2_dataprep.sh"
echo "=========================================================="
