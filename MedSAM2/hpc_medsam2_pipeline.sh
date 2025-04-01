#!/bin/bash
#SBATCH --job-name=MedSAM2_Pipeline
#SBATCH --output=MedSAM2_Pipeline_%j.log
#SBATCH --error=MedSAM2_Pipeline_%j.log
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=700G
#SBATCH --gres=gpu:4
#SBATCH --partition=main

###############################################################################
# MedSAM2 Complete Pipeline Script for HPC
# This script runs the entire MedSAM2 pipeline for Mpox lesion segmentation:
# 1. Environment setup (using venv)
# 2. Data preparation
# 3. Inference with pretrained model
# 4. Fine-tuning (if masks available)
# 5. Inference with fine-tuned model
###############################################################################

# 1) Print start time and node info
echo "=========================================================="
echo "Starting MedSAM2 Complete Pipeline on HPC"
echo "Current time: $(date)"
echo "Running on node: $(hostname)"
echo "GPU information:"
nvidia-smi
echo "Current directory: $(pwd)"
echo "=========================================================="

# 2) Define directories and settings
export BASE_DIR=${SCRATCH:-$HOME}/MedSAM2_Mpox
export MEDSAM_DIR=${BASE_DIR}/MedSAM2
export CHECKPOINT_DIR=${BASE_DIR}/checkpoints
export MPOX_DATA_DIR=${BASE_DIR}/mpox_data
export SCRIPTS_DIR=${BASE_DIR}/scripts
export RESULTS_DIR=${BASE_DIR}/results
export FINETUNE_DIR=${BASE_DIR}/finetune
export ENV_PATH=${BASE_DIR}/medsam2_env

# 3) Parse command line arguments
DO_SETUP=1
DO_DATAPREP=1
DO_INFERENCE=1
DO_FINETUNE=1
IMAGE_DIR=""
MASK_DIR=""
RUN_NAME=$(date +"%Y%m%d_%H%M%S")

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --skip-setup)
        DO_SETUP=0
        shift
        ;;
        --skip-dataprep)
        DO_DATAPREP=0
        shift
        ;;
        --skip-inference)
        DO_INFERENCE=0
        shift
        ;;
        --skip-finetune)
        DO_FINETUNE=0
        shift
        ;;
        --image-dir)
        IMAGE_DIR="$2"
        shift
        shift
        ;;
        --mask-dir)
        MASK_DIR="$2"
        shift
        shift
        ;;
        --run-name)
        RUN_NAME="$2"
        shift
        shift
        ;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done

# Create run directory with timestamp
RUN_DIR=${BASE_DIR}/runs/${RUN_NAME}
mkdir -p ${RUN_DIR}
mkdir -p ${SCRIPTS_DIR}

# Create log directory
LOG_DIR=${RUN_DIR}/logs
mkdir -p ${LOG_DIR}

# Start logging
MASTER_LOG=${LOG_DIR}/master_log.txt
exec > >(tee -a "${MASTER_LOG}") 2>&1

# 4) Create environment activation script if it doesn't exist
if [ ! -f "${BASE_DIR}/activate_env.sh" ]; then
    mkdir -p ${BASE_DIR}
    cat > ${BASE_DIR}/activate_env.sh << EOF
#!/bin/bash
# Helper script to activate MedSAM2 environment with venv

# Load modules (keep any modules you need from your HPC)
module purge
module load python/3.10
module load cuda/12.1  # Adjust as needed for your HPC
module load cudnn/8.9.5-cuda12  # Adjust as needed

# Activate venv environment
source ${ENV_PATH}/bin/activate

# Set CUDA_HOME
export CUDA_HOME=/usr/local/cuda-12.1  # Adjust as needed

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
fi

# 5) Create utility functions script if it doesn't exist
if [ ! -f "${BASE_DIR}/utils.sh" ]; then
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
fi

# 6) Environment setup
if [ ${DO_SETUP} -eq 1 ]; then
    echo "=========================================================="
    echo "STEP 1: Setting up MedSAM2 environment"
    echo "=========================================================="
    
    # Create the setup script
    cat > ${SCRIPTS_DIR}/hpc_medsam2_setup.sh << 'EOF'
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
# MedSAM2 Environment Setup Script for HPC (using venv)
###############################################################################

# 1) Print start time and node info
echo "=========================================================="
echo "Starting MedSAM2 Environment Setup on HPC (using venv)"
echo "Current time: $(date)"
echo "Running on node: $(hostname)"
echo "Current directory: $(pwd)"
echo "=========================================================="

# 2) Define directories and settings
BASE_DIR=${SCRATCH:-$HOME}/MedSAM2_Mpox
ENV_PATH=${BASE_DIR}/medsam2_env
MEDSAM_DIR=${BASE_DIR}/MedSAM2
CHECKPOINT_DIR=${BASE_DIR}/checkpoints
MPOX_DATA_DIR=${BASE_DIR}/mpox_data

mkdir -p ${BASE_DIR}
mkdir -p ${CHECKPOINT_DIR}

# 3) Load modules
echo "Loading required modules..."
module purge
module load python/3.10
module load cuda/12.1  # Adjust as needed for your HPC
module load cudnn/8.9.5-cuda12  # Adjust as needed

# 4) Create and activate Python venv environment
echo "Creating Python venv environment at ${ENV_PATH}..."
python3 -m venv ${ENV_PATH}

# Activate the venv
source ${ENV_PATH}/bin/activate

# Check if environment was successfully activated
if [[ $? -ne 0 ]]; then
    echo "ERROR: Failed to activate venv environment."
    exit 1
fi

# 5) Install PyTorch and dependencies
echo "Installing PyTorch 2.3.1 with CUDA support..."
pip install --upgrade pip
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
export CUDA_HOME=/usr/local/cuda-12.1  # Adjust for your HPC
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

# 12) Print summary and instructions
echo "=========================================================="
echo "MedSAM2 ENVIRONMENT SETUP COMPLETE (using venv)"
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
EOF
    chmod +x ${SCRIPTS_DIR}/hpc_medsam2_setup.sh
    
    # Run setup script and save log
    echo "Running environment setup script..."
    ${SCRIPTS_DIR}/hpc_medsam2_setup.sh 2>&1 | tee ${LOG_DIR}/setup_log.txt
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Environment setup failed. Check ${LOG_DIR}/setup_log.txt for details."
        exit 1
    fi
    
    echo "Environment setup completed successfully."
fi

# Source the environment activation script
echo "Activating MedSAM2 environment..."
source ${BASE_DIR}/activate_env.sh

# 7) Data preparation
if [ ${DO_DATAPREP} -eq 1 ]; then
    echo "=========================================================="
    echo "STEP 2: Preparing Mpox data"
    echo "=========================================================="
    
    # Check if image directory is provided
    if [ -z "${IMAGE_DIR}" ]; then
        IMAGE_DIR=${MPOX_DATA_DIR}/images
        echo "No image directory provided. Using default: ${IMAGE_DIR}"
    fi
    
    # Check if mask directory is provided
    if [ -z "${MASK_DIR}" ]; then
        MASK_DIR=${MPOX_DATA_DIR}/masks
        echo "No mask directory provided. Using default: ${MASK_DIR}"
    fi
    
    # Create image and mask directories if they don't exist
    mkdir -p ${IMAGE_DIR}
    mkdir -p ${MASK_DIR}
    
    # Check if images exist
    if [ ! -d "${IMAGE_DIR}" ] || [ -z "$(ls -A ${IMAGE_DIR} 2>/dev/null)" ]; then
        echo "WARNING: No images found in ${IMAGE_DIR}."
        echo "Please add your Mpox images to ${IMAGE_DIR} before running data preparation."
        
        # Ask if user wants to continue without images
        echo "Do you want to skip data preparation and continue with the rest of the pipeline? [y/N]"
        read -t 30 continue_without_images
        
        if [[ "${continue_without_images}" != "y" && "${continue_without_images}" != "Y" ]]; then
            echo "Exiting. Please add images and run the pipeline again."
            exit 1
        else
            echo "Continuing without data preparation..."
            DO_DATAPREP=0
        fi
    fi
    
    if [ ${DO_DATAPREP} -eq 1 ]; then
        # Copy the data preparation script to scripts directory
        cp hpc_medsam2_dataprep.sh ${SCRIPTS_DIR}/
        chmod +x ${SCRIPTS_DIR}/hpc_medsam2_dataprep.sh
        
        # Run data preparation script and save log
        echo "Running data preparation script..."
        ${SCRIPTS_DIR}/hpc_medsam2_dataprep.sh 2>&1 | tee ${LOG_DIR}/dataprep_log.txt
        
        if [ $? -ne 0 ]; then
            echo "ERROR: Data preparation failed. Check ${LOG_DIR}/dataprep_log.txt for details."
            exit 1
        fi
        
        echo "Data preparation completed successfully."
    fi
fi

# 8) Inference with pretrained model
if [ ${DO_INFERENCE} -eq 1 ]; then
    echo "=========================================================="
    echo "STEP 3: Running inference with pretrained MedSAM2"
    echo "=========================================================="
    
    # Check if preprocessed data exists
    NPZ_DIR=${MPOX_DATA_DIR}/npz_inference
    
    if [ ! -d "${NPZ_DIR}" ] || [ -z "$(ls -A ${NPZ_DIR} 2>/dev/null)" ]; then
        echo "ERROR: No preprocessed data found in ${NPZ_DIR}."
        
        if [ ${DO_DATAPREP} -eq 0 ]; then
            echo "You skipped data preparation. Please run data preparation first or use --skip-inference option."
            exit 1
        else
            echo "Data preparation was run but no data was produced. Check logs for errors."
            exit 1
        fi
    fi
    
    # Copy the inference script to scripts directory
    cp hpc_medsam2_inference.sh ${SCRIPTS_DIR}/
    chmod +x ${SCRIPTS_DIR}/hpc_medsam2_inference.sh
    
    # Run inference script and save log
    echo "Running inference script with pretrained model..."
    ${SCRIPTS_DIR}/hpc_medsam2_inference.sh 2>&1 | tee ${LOG_DIR}/inference_log.txt
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Inference failed. Check ${LOG_DIR}/inference_log.txt for details."
        exit 1
    fi
    
    echo "Inference with pretrained model completed successfully."
fi

# 9) Fine-tuning
if [ ${DO_FINETUNE} -eq 1 ]; then
    echo "=========================================================="
    echo "STEP 4: Fine-tuning MedSAM2 on Mpox data"
    echo "=========================================================="
    
    # Check if NPY data exists for fine-tuning
    NPY_DIR=${MPOX_DATA_DIR}/npy
    
    if [ ! -d "${NPY_DIR}" ] || [ -z "$(ls -A ${NPY_DIR} 2>/dev/null)" ]; then
        echo "WARNING: No training data found in ${NPY_DIR}."
        
        # Check if masks exist
        if [ -d "${MASK_DIR}" ] && [ ! -z "$(ls -A ${MASK_DIR} 2>/dev/null)" ]; then
            echo "Mask images found, but training data was not prepared correctly."
            echo "This could be due to an error in data preparation."
            echo "Skipping fine-tuning."
        else
            echo "No mask images found in ${MASK_DIR}."
            echo "Fine-tuning requires mask images for training."
            echo "Please add mask images to ${MASK_DIR} and run the pipeline again with --skip-setup --skip-inference options."
            echo "Skipping fine-tuning."
        fi
        
        DO_FINETUNE=0
    fi
    
    if [ ${DO_FINETUNE} -eq 1 ]; then
        # Copy the fine-tuning script to scripts directory
        cp hpc_medsam2_finetune.sh ${SCRIPTS_DIR}/
        chmod +x ${SCRIPTS_DIR}/hpc_medsam2_finetune.sh
        
        # Run fine-tuning script and save log
        echo "Running fine-tuning script..."
        ${SCRIPTS_DIR}/hpc_medsam2_finetune.sh 2>&1 | tee ${LOG_DIR}/finetune_log.txt
        
        if [ $? -ne 0 ]; then
            echo "ERROR: Fine-tuning failed. Check ${LOG_DIR}/finetune_log.txt for details."
            exit 1
        fi
        
        echo "Fine-tuning completed successfully."
    fi
fi

# 10) Print summary and next steps
echo "=========================================================="
echo "MEDSAM2 COMPLETE PIPELINE FINISHED"
echo "=========================================================="
echo "Run directory: ${RUN_DIR}"
echo "Logs: ${LOG_DIR}"
echo ""

# Check which steps were executed
steps_executed=""
if [ ${DO_SETUP} -eq 1 ]; then steps_executed="${steps_executed} Setup"; fi
if [ ${DO_DATAPREP} -eq 1 ]; then steps_executed="${steps_executed} DataPrep"; fi
if [ ${DO_INFERENCE} -eq 1 ]; then steps_executed="${steps_executed} Inference"; fi
if [ ${DO_FINETUNE} -eq 1 ]; then steps_executed="${steps_executed} Finetune"; fi

echo "Steps executed:${steps_executed}"
echo ""

# Create a symlink to the latest run
LATEST_LINK=${BASE_DIR}/latest_run
rm -f ${LATEST_LINK} 2>/dev/null
ln -s ${RUN_DIR} ${LATEST_LINK}
echo "Created symbolic link to the latest run: ${LATEST_LINK}"
echo ""

echo "Next steps:"
echo "1. Review the logs in ${LOG_DIR}"
echo "2. Check the results of inference and fine-tuning (if run)"
echo "3. Use the fine-tuned model for inference on new images (if fine-tuning was run)"
echo "=========================================================="

# Record end time
echo "Pipeline completed at: $(date)"
