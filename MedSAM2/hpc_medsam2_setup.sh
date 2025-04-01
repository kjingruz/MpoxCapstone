#!/bin/bash
#SBATCH --job-name=MedSAM2_Setup
#SBATCH --output=MedSAM2_Setup_%j.log
#SBATCH --error=MedSAM2_Setup_%j.log
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=main

###############################################################################
# MedSAM2 Environment Setup Script for HPC
###############################################################################

# 1) Print start time and node info
echo "=========================================================="
echo "Starting MedSAM2 Environment Setup on HPC"
echo "Current time: $(date)"
echo "Running on node: $(hostname)"
echo "Current directory: $(pwd)"
echo "=========================================================="

# 2) Define directories and settings
BASE_DIR=${HOME}/Mpox
ENV_NAME=sam2_in_med
MEDSAM_DIR=${BASE_DIR}/MedSAM2
CHECKPOINT_DIR=${BASE_DIR}/checkpoints
MPOX_DATA_DIR=${BASE_DIR}/data
SCRIPTS_DIR=${BASE_DIR}/scripts

mkdir -p ${BASE_DIR}
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${SCRIPTS_DIR}

# 3) Load modules
echo "Loading required modules..."
module purge
module load python/3.10
module load anaconda3
module load cuda/12.1
module load cudnn/8.9.5-cuda12

# 4) Create and activate conda environment
echo "Creating conda environment ${ENV_NAME}..."
conda create -n ${ENV_NAME} python=3.10 -y

# Set conda environment activation command
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# Check if environment was successfully activated
if [[ $? -ne 0 ]]; then
    echo "ERROR: Failed to activate conda environment."
    exit 1
fi

# 5) Install PyTorch and dependencies
echo "Installing PyTorch 2.3.1 with CUDA support..."
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

echo "Installing additional dependencies..."
pip install matplotlib scipy scikit-image opencv-python tqdm nibabel pycocotools
pip install pandas seaborn tensorboard

# 6) Clone SAM2 repository 
echo "Cloning SAM2 repository..."
cd ${BASE_DIR}
if [ -d "${MEDSAM_DIR}" ]; then
    echo "SAM2 repository already exists. Updating..."
    cd ${MEDSAM_DIR}
    git pull
else
    git clone https://github.com/facebookresearch/sam2.git ${MEDSAM_DIR}
    cd ${MEDSAM_DIR}
fi

# 7) Set CUDA_HOME environment variable
export CUDA_HOME=/usr/local/cuda-12.1
echo "Setting CUDA_HOME=${CUDA_HOME}"

# 8) Install SAM2 package in development mode
echo "Installing SAM2 package..."
pip install -e .

# 9) Download SAM2 checkpoints
echo "Downloading SAM2 checkpoints..."
cd ${CHECKPOINT_DIR}

# Using the latest SAM 2.1 models
BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
checkpoints=(
    "sam2.1_hiera_tiny.pt" 
    "sam2.1_hiera_base_plus.pt"
)

# Download only base_plus checkpoint by default since we have sufficient resources
for ckpt in "${checkpoints[@]}"; do
    if [[ $ckpt == "sam2.1_hiera_base_plus.pt" ]]; then
        if [ ! -f "${CHECKPOINT_DIR}/${ckpt}" ]; then
            echo "Downloading ${ckpt}..."
            wget ${BASE_URL}${ckpt}
        else
            echo "${ckpt} already exists."
        fi
    fi
done

# 10) Copy Python scripts to scripts directory
echo "Copying Python scripts..."
cp ${BASE_DIR}/coco_to_masks.py ${SCRIPTS_DIR}/
cp ${BASE_DIR}/mpox_data_prep.py ${SCRIPTS_DIR}/
cp ${BASE_DIR}/finetune_medsam2_mpox.py ${SCRIPTS_DIR}/
cp ${BASE_DIR}/run_medsam2_inference.py ${SCRIPTS_DIR}/
cp ${BASE_DIR}/evaluate_medsam2.py ${SCRIPTS_DIR}/

# Make scripts executable
chmod +x ${SCRIPTS_DIR}/*.py

# 11) Create a helper script to activate the environment
cat > ${BASE_DIR}/activate_env.sh << EOF
#!/bin/bash
# Helper script to activate MedSAM2 environment

# Load modules
module purge
module load python/3.10
module load anaconda3
module load cuda/12.1
module load cudnn/8.9.5-cuda12

# Activate conda environment
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# Set CUDA_HOME
export CUDA_HOME=/usr/local/cuda-12.1

# Print environment info
echo "MedSAM2 environment activated."
echo "Python: \$(which python)"
echo "PyTorch: \$(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: \$(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Using CUDA_HOME=\${CUDA_HOME}"

# Set MedSAM2 base directory
export MEDSAM2_BASE="${BASE_DIR}"
export MEDSAM2_CHECKPOINTS="${CHECKPOINT_DIR}"
export MEDSAM2_DATA="${MPOX_DATA_DIR}"
EOF
chmod +x ${BASE_DIR}/activate_env.sh

# 12) Print summary and instructions
echo "=========================================================="
echo "MedSAM2 ENVIRONMENT SETUP COMPLETE"
echo "=========================================================="
echo "Base directory: ${BASE_DIR}"
echo "SAM2 code: ${MEDSAM_DIR}"
echo "Checkpoints: ${CHECKPOINT_DIR}"
echo "Mpox data: ${MPOX_DATA_DIR}"
echo ""
echo "To activate the environment in future scripts, source the activation script:"
echo "source ${BASE_DIR}/activate_env.sh"
echo ""
echo "Next steps:"
echo "1. Convert COCO annotations to masks: sbatch hpc_medsam2_coco_to_mask.sh"
echo "2. Run the data preparation script: sbatch hpc_medsam2_dataprep.sh"
echo "3. Run fine-tuning: sbatch hpc_medsam2_finetune.sh"
echo "4. Run inference: sbatch hpc_medsam2_inference.sh"
echo "=========================================================="
