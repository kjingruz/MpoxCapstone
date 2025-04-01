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
BASE_DIR=${SCRATCH:-$HOME}/MedSAM2_Mpox
ENV_NAME=sam2_in_med
MEDSAM_DIR=${BASE_DIR}/MedSAM2
CHECKPOINT_DIR=${BASE_DIR}/checkpoints
MPOX_DATA_DIR=${BASE_DIR}/mpox_data

mkdir -p ${BASE_DIR}
mkdir -p ${CHECKPOINT_DIR}

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
pip install matplotlib scipy scikit-image opencv-python tqdm nibabel gradio==3.38.0 tensorboard

# 6) Clone MedSAM2 repository
echo "Cloning MedSAM2 repository..."
cd ${BASE_DIR}
if [ -d "${MEDSAM_DIR}" ]; then
    echo "MedSAM2 repository already exists. Updating..."
    cd ${MEDSAM_DIR}
    git checkout MedSAM2
    git pull
else
    git clone -b MedSAM2 https://github.com/bowang-lab/MedSAM/ ${MEDSAM_DIR}
    cd ${MEDSAM_DIR}
fi

# 7) Set CUDA_HOME environment variable
export CUDA_HOME=/usr/local/cuda-12.1
echo "Setting CUDA_HOME=${CUDA_HOME}"

# 8) Install MedSAM2 package in development mode
echo "Installing MedSAM2 package..."
pip install -e .

# 9) Download SAM2 checkpoints
echo "Downloading SAM2 checkpoints..."
cd ${CHECKPOINT_DIR}

BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/072824/"
checkpoints=(
    "sam2_hiera_tiny.pt" 
    "sam2_hiera_small.pt" 
    "sam2_hiera_base_plus.pt"
)

# Download only base_plus checkpoint by default to save time
# (uncomment others if needed)
for ckpt in "${checkpoints[@]}"; do
    if [[ $ckpt == "sam2_hiera_base_plus.pt" ]]; then
        if [ ! -f "${CHECKPOINT_DIR}/${ckpt}" ]; then
            echo "Downloading ${ckpt}..."
            wget ${BASE_URL}${ckpt}
        else
            echo "${ckpt} already exists."
        fi
    fi
done

# 10) Download MedSAM2 pretrained weights
echo "Downloading MedSAM2 pretrained weights..."
if [ ! -f "${CHECKPOINT_DIR}/MedSAM2_pretrain.pth" ]; then
    wget -O ${CHECKPOINT_DIR}/MedSAM2_pretrain.pth \
        https://huggingface.co/jiayuanz3/MedSAM2_pretrain/resolve/main/MedSAM2_pretrain.pth
else
    echo "MedSAM2 pretrained weights already exist."
fi

# 11) Create directory structure for Mpox data
echo "Creating directory structure for Mpox data..."
mkdir -p ${MPOX_DATA_DIR}/{images,masks,npz_inference,npz_train,npz_val,npy}

# 12) Create helper utility for copying scripts
cat > ${BASE_DIR}/utils.sh << 'EOF'
#!/bin/bash
# Helper functions for MedSAM2 scripts

function copy_scripts() {
    source_dir="$1"
    target_dir="$2"
    
    mkdir -p "$target_dir"
    
    # Copy Python scripts
    if [ -d "$source_dir" ]; then
        cp "$source_dir"/*.py "$target_dir"/ 2>/dev/null || true
        cp "$source_dir"/*.sh "$target_dir"/ 2>/dev/null || true
        echo "Copied scripts from $source_dir to $target_dir"
    else
        echo "Source directory $source_dir does not exist"
    fi
    
    # Make shell scripts executable
    chmod +x "$target_dir"/*.sh 2>/dev/null || true
}
EOF
chmod +x ${BASE_DIR}/utils.sh

# 13) Create a helper script to activate the environment
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

# 14) Print summary and instructions
echo "=========================================================="
echo "MedSAM2 ENVIRONMENT SETUP COMPLETE"
echo "=========================================================="
echo "Base directory: ${BASE_DIR}"
echo "MedSAM2 code: ${MEDSAM_DIR}"
echo "Checkpoints: ${CHECKPOINT_DIR}"
echo "Mpox data: ${MPOX_DATA_DIR}"
echo ""
echo "To activate the environment in future scripts, source the activation script:"
echo "source ${BASE_DIR}/activate_env.sh"
echo ""
echo "Next steps:"
echo "1. Upload your Mpox images to ${MPOX_DATA_DIR}/images"
echo "2. If available, upload your Mpox masks to ${MPOX_DATA_DIR}/masks"
echo "3. Run the data preparation script: sbatch hpc_medsam2_dataprep.sh"
echo "4. Run inference: sbatch hpc_medsam2_inference.sh"
echo "5. (Optional) Fine-tune the model: sbatch hpc_medsam2_finetune.sh"
echo "=========================================================="
