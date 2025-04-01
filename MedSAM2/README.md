# MedSAM2 Implementation for Mpox Lesion Segmentation

This repository contains a complete implementation of MedSAM2 for Mpox lesion segmentation, including environment setup, data preparation, inference, fine-tuning, and evaluation. The implementation follows the official MedSAM2 repository structure from the [bowang-lab](https://github.com/bowang-lab/MedSAM/tree/MedSAM2), optimized for HPC environments.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Model Selection](#model-selection)
- [HPC Integration](#hpc-integration)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## Overview

MedSAM2 (Medical Segment Anything 2) is a state-of-the-art medical image segmentation model based on Meta AI's Segment Anything Model 2 (SAM2). This implementation is specifically designed for segmenting Mpox lesions, with minimal prompting and the ability to fine-tune on custom data.

## Key Features

- **Complete Pipeline**: Environment setup, data preparation, inference, fine-tuning, and evaluation
- **HPC Integration**: SLURM batch scripts optimized for multi-GPU training
- **Few-shot Learning**: Works with minimal labeled data
- **Interactive Prompting**: Box or point-based prompting for accurate segmentation
- **Evaluation Tools**: Comprehensive metrics and visualization for model performance

## Repository Structure

```
MedSAM2_Mpox/
├── scripts/                     # Python scripts
│   ├── mpox_data_prep.py        # Data preparation script
│   ├── run_medsam2_inference.py # Inference script
│   ├── run_medsam2_finetune.py  # Fine-tuning script
│   └── evaluate_medsam2.py      # Evaluation script
├── hpc_scripts/                 # HPC batch scripts
│   ├── hpc_medsam2_setup.sh     # Environment setup
│   ├── hpc_medsam2_dataprep.sh  # Data preparation
│   ├── hpc_medsam2_inference.sh # Inference
│   ├── hpc_medsam2_finetune.sh  # Fine-tuning
│   └── hpc_medsam2_pipeline.sh  # Complete pipeline
├── MedSAM2/                     # Official MedSAM2 repository (cloned)
├── checkpoints/                 # Model checkpoints
│   ├── sam2_hiera_base_plus.pt  # Base SAM2 model
│   └── MedSAM2_pretrain.pth     # MedSAM2 pretrained weights
├── mpox_data/                   # Mpox data
│   ├── images/                  # Original images
│   ├── masks/                   # Mask images (for training)
│   ├── npz_inference/           # Preprocessed data for inference
│   ├── npz_train/               # Preprocessed data for training
│   ├── npz_val/                 # Preprocessed data for validation
│   └── npy/                     # Preprocessed data in NPY format
├── results/                     # Inference results
├── finetune/                    # Fine-tuning results
├── latest_run/                  # Symlink to latest run
└── activate_env.sh              # Environment activation script
```

## Installation and Setup

### Prerequisites

- Python 3.10+
- CUDA 12.1+ (for GPU acceleration)
- Conda or similar environment manager

### Automated Setup

Run the complete setup script:

```bash
# Single command setup
sbatch hpc_medsam2_setup.sh
```

### Manual Setup

1. **Clone this repository**:
```bash
git clone <repository-url> MedSAM2_Mpox
cd MedSAM2_Mpox
```

2. **Create and activate conda environment**:
```bash
conda create -n sam2_in_med python=3.10 -y
conda activate sam2_in_med
```

3. **Install dependencies**:
```bash
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install matplotlib scipy scikit-image opencv-python tqdm nibabel
pip install gradio==3.38.0
```

4. **Clone MedSAM2 repository**:
```bash
git clone -b MedSAM2 https://github.com/bowang-lab/MedSAM/ MedSAM2
cd MedSAM2
pip install -e .
cd ..
```

5. **Download model checkpoints**:
```bash
mkdir -p checkpoints
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt
wget https://huggingface.co/jiayuanz3/MedSAM2_pretrain/resolve/main/MedSAM2_pretrain.pth
cd ..
```

## Usage

### Complete Pipeline

Run the entire pipeline with a single command:

```bash
sbatch hpc_medsam2_pipeline.sh
```

You can customize the pipeline with the following options:
- `--skip-setup`: Skip environment setup
- `--skip-dataprep`: Skip data preparation
- `--skip-inference`: Skip inference with pretrained model
- `--skip-finetune`: Skip fine-tuning
- `--image-dir PATH`: Specify custom image directory
- `--mask-dir PATH`: Specify custom mask directory
- `--run-name NAME`: Specify a custom run name

### Individual Steps

#### 1. Data Preparation

```bash
# For inference only (no masks)
sbatch hpc_medsam2_dataprep.sh

# For training/fine-tuning (with masks)
# Add your images to mpox_data/images/
# Add your masks to mpox_data/masks/
sbatch hpc_medsam2_dataprep.sh
```

#### 2. Inference with Pretrained Model

```bash
sbatch hpc_medsam2_inference.sh
```

#### 3. Fine-tuning

```bash
sbatch hpc_medsam2_finetune.sh
```

#### 4. Evaluation

```bash
# Compare ground truth with predictions
python evaluate_medsam2.py \
    --pred_dir results/inference_XXX/masks \
    --gt_dir mpox_data/masks \
    --output_dir evaluation_results

# Compare multiple models
python evaluate_medsam2.py \
    --pred_dir results/inference_pretrained/masks \
    --gt_dir mpox_data/masks \
    --compare results/inference_finetuned/masks \
    --model_names "Pretrained" "Fine-tuned" \
    --output_dir model_comparison
```

## Model Selection

MedSAM2 supports four model sizes:

| Model | Parameters | GPU Memory | Inference Speed | Recommendation |
|-------|------------|------------|-----------------|----------------|
| tiny | ~10M | ~2GB | ~0.1s/image | CPU or entry-level GPU |
| small | ~40M | ~4GB | ~0.2s/image | Mid-range GPU |
| base_plus | ~90M | ~8GB | ~0.3s/image | Good GPU (RTX 3060+) |
| large | ~300M | ~16GB | ~0.5s/image | High-end GPU (RTX 4070+) |

To change the model size, modify the `sam2_checkpoint` parameter in the scripts or use the config file.

## HPC Integration

This implementation is optimized for HPC environments with SLURM. Key features:

- **Multi-GPU Training**: Efficient fine-tuning using multiple GPUs
- **Job Dependencies**: Automatically handles dependencies between steps
- **Resource Allocation**: Appropriate CPU, memory, and GPU allocation
- **Logging**: Comprehensive logging for each step
- **Error Handling**: Robust error detection and reporting

### Custom HPC Configuration

You can customize the SLURM directives in each script:

```bash
#SBATCH --job-name=MedSAM2_XXX
#SBATCH --output=MedSAM2_XXX_%j.log
#SBATCH --error=MedSAM2_XXX_%j.log
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=700G
#SBATCH --gres=gpu:4
#SBATCH --partition=main
```

## Few-Shot Learning Approach

For datasets with limited labeled examples:

1. Add your few labeled examples to `mpox_data/images/` and `mpox_data/masks/`
2. Run data preparation: `sbatch hpc_medsam2_dataprep.sh`
3. Fine-tune with modified learning rate and epochs:
   ```bash
   # Edit hpc_medsam2_finetune.sh to use:
   # --batch_size 4 --num_epochs 20 --learning_rate 5e-6
   sbatch hpc_medsam2_finetune.sh
   ```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Use a smaller model (tiny or small)
   - Reduce batch size in fine-tuning
   - Use fewer GPUs with larger memory per GPU

2. **Installation Errors**:
   - Check CUDA compatibility with PyTorch version
   - Verify CUDA_HOME is set correctly

3. **Segmentation Quality Issues**:
   - Try different prompting methods (box vs. points)
   - Adjust bbox_shift parameter for box prompting
   - Fine-tune on a small set of Mpox images

4. **HPC-specific Issues**:
   - Verify module availability (Python, CUDA, cuDNN)
   - Check job resource allocation (memory, GPUs)
   - Review logs for specific errors

### Support

For additional support:
- Check the [official MedSAM2 repository](https://github.com/bowang-lab/MedSAM/tree/MedSAM2)
- Review the [SAM2 documentation](https://github.com/facebookresearch/segment-anything-2)
- Submit issues to this repository

## References

- **MedSAM2 Paper**: [Segment Anything in Medical Images and Videos: Benchmark and Deployment](https://arxiv.org/abs/2408.03322)
- **Official MedSAM2 Repository**: [https://github.com/bowang-lab/MedSAM/tree/MedSAM2](https://github.com/bowang-lab/MedSAM/tree/MedSAM2)
- **SAM2 Repository**: [https://github.com/facebookresearch/segment-anything-2](https://github.com/facebookresearch/segment-anything-2)
