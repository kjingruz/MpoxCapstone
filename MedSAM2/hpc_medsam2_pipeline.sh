#!/bin/bash
#SBATCH --job-name=MedSAM2_Pipeline
#SBATCH --output=MedSAM2_Pipeline_%j.log
#SBATCH --error=MedSAM2_Pipeline_%j.log
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=700G
#SBATCH --gres=gpu:4
#SBATCH --partition=main

###############################################################################
# MedSAM2 Complete Pipeline Script for HPC
# This script runs the entire MedSAM2 workflow for Mpox lesion segmentation:
# 1. Environment setup
# 2. COCO to mask conversion
# 3. Data preparation
# 4. Fine-tuning
# 5. Inference
# 6. Evaluation
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
BASE_DIR=${HOME}/Mpox
MPOX_DIR=${BASE_DIR}/data/Monkey_Pox
COCO_FILE=${MPOX_DIR}/annotation/instances_default.json
SCRIPTS_DIR=${BASE_DIR}/scripts
CHECKPOINT_DIR=${BASE_DIR}/checkpoints
MPOX_DATA_DIR=${BASE_DIR}/mpox_data
FINETUNE_DIR=${BASE_DIR}/finetune
RESULTS_DIR=${BASE_DIR}/results
EVALUATION_DIR=${BASE_DIR}/evaluation

# 3) Parse command line arguments
DO_SETUP=1
DO_COCO=1
DO_DATAPREP=1
DO_FINETUNE=1
DO_INFERENCE=1
DO_EVALUATE=1
RUN_NAME=$(date +"%Y%m%d_%H%M%S")

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --skip-setup)
        DO_SETUP=0
        shift
        ;;
        --skip-coco)
        DO_COCO=0
        shift
        ;;
        --skip-dataprep)
        DO_DATAPREP=0
        shift
        ;;
        --skip-finetune)
        DO_FINETUNE=0
        shift
        ;;
        --skip-inference)
        DO_INFERENCE=0
        shift
        ;;
        --skip-evaluate)
        DO_EVALUATE=0
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

# Create log directory
LOG_DIR=${RUN_DIR}/logs
mkdir -p ${LOG_DIR}

# Start logging
MASTER_LOG=${LOG_DIR}/master_log.txt
exec > >(tee -a "${MASTER_LOG}") 2>&1

# 4) Environment setup
if [ ${DO_SETUP} -eq 1 ]; then
    echo "=========================================================="
    echo "STEP 1: Setting up MedSAM2 environment"
    echo "=========================================================="
    
    SETUP_LOG=${LOG_DIR}/setup_log.txt
    bash ${BASE_DIR}/hpc_medsam2_setup.sh 2>&1 | tee ${SETUP_LOG}
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "ERROR: Environment setup failed. Check ${SETUP_LOG} for details."
        exit 1
    fi
fi

# Source the environment activation script
if [ -f "${BASE_DIR}/activate_env.sh" ]; then
    echo "Activating MedSAM2 environment..."
    source ${BASE_DIR}/activate_env.sh
else
    echo "ERROR: Environment activation script not found at ${BASE_DIR}/activate_env.sh"
    echo "Please run the setup step first."
    exit 1
fi

# 5) COCO to mask conversion
if [ ${DO_COCO} -eq 1 ]; then
    echo "=========================================================="
    echo "STEP 2: Converting COCO annotations to masks"
    echo "=========================================================="
    
    COCO_LOG=${LOG_DIR}/coco_log.txt
    
    # Check if COCO file exists
    if [ ! -f "${COCO_FILE}" ]; then
        echo "ERROR: COCO annotation file not found at ${COCO_FILE}."
        exit 1
    fi
    
    # Run COCO to mask conversion
    python ${SCRIPTS_DIR}/coco_to_masks_simple.py \
        --coco_json ${COCO_FILE} \
        --img_dir ${MPOX_DIR} \
        --output_dir ${MPOX_DIR}/masks \
        --visualize 2>&1 | tee ${COCO_LOG}
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "ERROR: COCO to mask conversion failed. Check ${COCO_LOG} for details."
        exit 1
    fi
fi

# 6) Data preparation
if [ ${DO_DATAPREP} -eq 1 ]; then
    echo "=========================================================="
    echo "STEP 3: Preparing Mpox data"
    echo "=========================================================="
    
    DATAPREP_LOG=${LOG_DIR}/dataprep_log.txt
    
    # Check if image and mask directories exist
    if [ ! -d "${MPOX_DIR}" ]; then
        echo "ERROR: Mpox image directory not found at ${MPOX_DIR}."
        exit 1
    fi
    
    if [ ! -d "${MPOX_DIR}/masks" ]; then
        echo "ERROR: Masks directory not found at ${MPOX_DIR}/masks."
        echo "Please run the COCO to mask conversion step first."
        exit 1
    fi
    
    # Run data preparation
    python ${SCRIPTS_DIR}/mpox_data_prep.py \
        --image_dir ${MPOX_DIR} \
        --mask_dir ${MPOX_DIR}/masks \
        --output_dir ${MPOX_DATA_DIR} \
        --mode training \
        --val_ratio 0.2 \
        --target_size 1024 1024 \
        --down_size 256 256 \
        --num_workers $(nproc) \
        --visualize 2>&1 | tee ${DATAPREP_LOG}
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "ERROR: Data preparation failed. Check ${DATAPREP_LOG} for details."
        exit 1
    fi
fi

# 7) Fine-tuning
if [ ${DO_FINETUNE} -eq 1 ]; then
    echo "=========================================================="
    echo "STEP 4: Fine-tuning SAM2 on Mpox data"
    echo "=========================================================="
    
    FINETUNE_LOG=${LOG_DIR}/finetune_log.txt
    FINETUNE_OUTPUT_DIR=${FINETUNE_DIR}/${RUN_NAME}
    mkdir -p ${FINETUNE_OUTPUT_DIR}
    
    # Check if training data exists
    if [ ! -d "${MPOX_DATA_DIR}/npy/imgs" ] || [ -z "$(ls -A ${MPOX_DATA_DIR}/npy/imgs 2>/dev/null)" ]; then
        echo "ERROR: No training data found in ${MPOX_DATA_DIR}/npy/imgs."
        echo "Please run the data preparation step first."
        exit 1
    fi
    
    # Check if checkpoint exists
    SAM2_CHECKPOINT=${CHECKPOINT_DIR}/sam2.1_hiera_base_plus.pt
    if [ ! -f "${SAM2_CHECKPOINT}" ]; then
        echo "ERROR: SAM2 checkpoint not found at ${SAM2_CHECKPOINT}."
        echo "Please download the checkpoint first."
        exit 1
    fi
    
    # Set batch size based on available GPU memory
    BATCH_SIZE=4
    if [[ $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1) -lt 16000 ]]; then
        BATCH_SIZE=4
        echo "Limited GPU memory detected, reducing batch size to ${BATCH_SIZE}"
    fi

    #ln -s /home/zhangk/Mpox/MedSAM2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml /home/zhangk/Mpox/MedSAM2/sam2/sam2.1_hiera_b+.yaml
    
    # Run fine-tuning
    python ${SCRIPTS_DIR}/finetune_medsam2_mpox.py \
        --data_dir ${MPOX_DATA_DIR}/npy \
        --output_dir ${FINETUNE_OUTPUT_DIR} \
        --sam2_checkpoint ${SAM2_CHECKPOINT} \
        --model_cfg "sam2.1/sam2.1_hiera_b+" \
        --batch_size ${BATCH_SIZE} \
        --num_epochs 30 \
        --learning_rate 1e-5 \
        --bbox_shift 10 \
        --device cuda \
        --num_workers 16 \
        --vis_samples 8 2>&1 | tee ${FINETUNE_LOG}
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "ERROR: Fine-tuning failed. Check ${FINETUNE_LOG} for details."
        exit 1
    fi
    
    # Copy the final model to a standard location
    FINAL_MODEL=${FINETUNE_OUTPUT_DIR}/medsam2_mpox_final.pth
    STANDARD_MODEL=${BASE_DIR}/medsam2_mpox.pth
    cp ${FINAL_MODEL} ${STANDARD_MODEL}
    
    # Create a symbolic link to the latest fine-tuning run
    LATEST_FINETUNE_LINK=${BASE_DIR}/latest_finetune
    rm -f ${LATEST_FINETUNE_LINK} 2>/dev/null
    ln -s ${FINETUNE_OUTPUT_DIR} ${LATEST_FINETUNE_LINK}
fi

# 8) Inference
if [ ${DO_INFERENCE} -eq 1 ]; then
    echo "=========================================================="
    echo "STEP 5: Running inference with SAM2 and fine-tuned models"
    echo "=========================================================="
    
    INFERENCE_LOG=${LOG_DIR}/inference_log.txt
    INFERENCE_OUTPUT_DIR=${RESULTS_DIR}/${RUN_NAME}
    mkdir -p ${INFERENCE_OUTPUT_DIR}
    
    # Check if inference data exists
    if [ ! -d "${MPOX_DATA_DIR}/npz_inference" ] || [ -z "$(ls -A ${MPOX_DATA_DIR}/npz_inference 2>/dev/null)" ]; then
        echo "ERROR: No inference data found in ${MPOX_DATA_DIR}/npz_inference."
        echo "Please run the data preparation step first."
        exit 1
    fi
    
    # Set model paths
    SAM2_CHECKPOINT=${CHECKPOINT_DIR}/sam2.1_hiera_base_plus.pt
    FINETUNED_MODEL=${BASE_DIR}/medsam2_mpox.pth
    
    # Run inference with base model
    BASE_OUTPUT_DIR=${INFERENCE_OUTPUT_DIR}/base_model
    mkdir -p ${BASE_OUTPUT_DIR}
    
    python ${SCRIPTS_DIR}/run_medsam2_inference.py \
        --input_dir ${MPOX_DATA_DIR}/npz_inference \
        --output_dir ${BASE_OUTPUT_DIR} \
        --sam2_checkpoint ${SAM2_CHECKPOINT} \
        --model_cfg "sam2.1_hiera_b+.yaml" \
        --prompt_method box \
        --bbox_shift 10 \
        --device cuda \
        --num_workers $(nproc) 2>&1 | tee ${INFERENCE_LOG}
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "ERROR: Base model inference failed. Check ${INFERENCE_LOG} for details."
        exit 1
    fi
    
    # Run inference with fine-tuned model (if available)
    if [ -f "${FINETUNED_MODEL}" ]; then
        FINETUNED_OUTPUT_DIR=${INFERENCE_OUTPUT_DIR}/finetuned_model
        mkdir -p ${FINETUNED_OUTPUT_DIR}
        
        python ${SCRIPTS_DIR}/run_medsam2_inference.py \
            --input_dir ${MPOX_DATA_DIR}/npz_inference \
            --output_dir ${FINETUNED_OUTPUT_DIR} \
            --sam2_checkpoint ${SAM2_CHECKPOINT} \
            --medsam2_checkpoint ${FINETUNED_MODEL} \
            --model_cfg "sam2.1_hiera_b+.yaml" \
            --prompt_method box \
            --bbox_shift 10 \
            --device cuda \
            --num_workers $(nproc) 2>&1 | tee -a ${INFERENCE_LOG}
        
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            echo "ERROR: Fine-tuned model inference failed. Check ${INFERENCE_LOG} for details."
            exit 1
        fi
    fi
    
    # Create a symbolic link to the latest inference run
    LATEST_INFERENCE_LINK=${BASE_DIR}/latest_inference
    rm -f ${LATEST_INFERENCE_LINK} 2>/dev/null
    ln -s ${INFERENCE_OUTPUT_DIR} ${LATEST_INFERENCE_LINK}
fi

# 9) Evaluation
if [ ${DO_EVALUATE} -eq 1 ]; then
    echo "=========================================================="
    echo "STEP 6: Evaluating model performance"
    echo "=========================================================="
    
    EVAL_LOG=${LOG_DIR}/evaluation_log.txt
    EVAL_OUTPUT_DIR=${EVALUATION_DIR}/${RUN_NAME}
    mkdir -p ${EVAL_OUTPUT_DIR}
    
    # Define prediction and ground truth directories
    INFERENCE_DIR=${RESULTS_DIR}/${RUN_NAME}
    BASE_PRED_DIR=${INFERENCE_DIR}/base_model/masks
    FINETUNED_PRED_DIR=${INFERENCE_DIR}/finetuned_model/masks
    GT_DIR=${MPOX_DATA_DIR}/npz_val
    
    # Check if prediction directories exist
    if [ ! -d "${BASE_PRED_DIR}" ]; then
        echo "ERROR: Base model predictions not found at ${BASE_PRED_DIR}."
        echo "Please run the inference step first."
        exit 1
    fi
    
    # Run evaluation for base model
    BASE_EVAL_DIR=${EVAL_OUTPUT_DIR}/base_model
    mkdir -p ${BASE_EVAL_DIR}
    
    python ${SCRIPTS_DIR}/evaluate_medsam2.py \
        --pred_dir ${BASE_PRED_DIR} \
        --gt_dir ${GT_DIR} \
        --output_dir ${BASE_EVAL_DIR} \
        --num_workers $(nproc) 2>&1 | tee ${EVAL_LOG}
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "ERROR: Base model evaluation failed. Check ${EVAL_LOG} for details."
        exit 1
    fi
    
    # Run evaluation for fine-tuned model and comparison (if available)
    if [ -d "${FINETUNED_PRED_DIR}" ]; then
        FINETUNED_EVAL_DIR=${EVAL_OUTPUT_DIR}/finetuned_model
        mkdir -p ${FINETUNED_EVAL_DIR}
        
        python ${SCRIPTS_DIR}/evaluate_medsam2.py \
            --pred_dir ${FINETUNED_PRED_DIR} \
            --gt_dir ${GT_DIR} \
            --output_dir ${FINETUNED_EVAL_DIR} \
            --num_workers $(nproc) 2>&1 | tee -a ${EVAL_LOG}
        
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            echo "ERROR: Fine-tuned model evaluation failed. Check ${EVAL_LOG} for details."
            exit 1
        fi
        
        # Compare the two models
        COMPARISON_DIR=${EVAL_OUTPUT_DIR}/comparison
        mkdir -p ${COMPARISON_DIR}
        
        python ${SCRIPTS_DIR}/evaluate_medsam2.py \
            --pred_dir ${BASE_PRED_DIR} \
            --gt_dir ${GT_DIR} \
            --output_dir ${COMPARISON_DIR} \
            --compare ${FINETUNED_PRED_DIR} \
            --model_names "SAM2-Base" "MedSAM2-Mpox" \
            --num_workers $(nproc) 2>&1 | tee -a ${EVAL_LOG}
        
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            echo "ERROR: Model comparison failed. Check ${EVAL_LOG} for details."
            exit 1
        fi
    fi
    
    # Create a symbolic link to the latest evaluation
    LATEST_EVAL_LINK=${BASE_DIR}/latest_evaluation
    rm -f ${LATEST_EVAL_LINK} 2>/dev/null
    ln -s ${EVAL_OUTPUT_DIR} ${LATEST_EVAL_LINK}
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
if [ ${DO_COCO} -eq 1 ]; then steps_executed="${steps_executed} COCO2Mask"; fi
if [ ${DO_DATAPREP} -eq 1 ]; then steps_executed="${steps_executed} DataPrep"; fi
if [ ${DO_FINETUNE} -eq 1 ]; then steps_executed="${steps_executed} Finetune"; fi
if [ ${DO_INFERENCE} -eq 1 ]; then steps_executed="${steps_executed} Inference"; fi
if [ ${DO_EVALUATE} -eq 1 ]; then steps_executed="${steps_executed} Evaluate"; fi

echo "Steps executed:${steps_executed}"
echo ""

# Create a symlink to the latest run
LATEST_RUN_LINK=${BASE_DIR}/latest_run
rm -f ${LATEST_RUN_LINK} 2>/dev/null
ln -s ${RUN_DIR} ${LATEST_RUN_LINK}
echo "Created symbolic link to the latest run: ${LATEST_RUN_LINK}"
echo ""

echo "Results Summary:"
echo "  - Fine-tuned model: ${BASE_DIR}/medsam2_mpox.pth"
echo "  - Inference results: ${RESULTS_DIR}/${RUN_NAME}"
echo "  - Evaluation results: ${EVALUATION_DIR}/${RUN_NAME}"
echo ""

echo "Next steps:"
echo "1. Review the evaluation metrics and visualizations"
echo "2. Use the fine-tuned model for inference on new images"
echo "3. Further improve the model with additional data if needed"
echo "=========================================================="

# Record end time
echo "Pipeline completed at: $(date)"
