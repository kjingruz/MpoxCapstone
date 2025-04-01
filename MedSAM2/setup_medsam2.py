#!/usr/bin/env python3
"""
Setup script for MedSAM2 environment and repository for Mpox lesion segmentation.
This script automates the process of setting up the MedSAM2 environment and downloading 
required models and code.
"""

import os
import subprocess
import sys
import argparse
from pathlib import Path
import shutil
import urllib.request

def run_command(command, description=None):
    """Run a shell command and print its output."""
    if description:
        print(f"\n=== {description} ===")
    
    print(f"Running: {command}")
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    
    # Print output in real-time
    for line in process.stdout:
        print(line.strip())
    
    # Wait for process to complete and get return code
    return_code = process.wait()
    if return_code != 0:
        print(f"Command failed with return code {return_code}")
        return False
    return True

def check_cuda():
    """Check if CUDA is available and which version."""
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print("CUDA is available. GPU information:")
            print(result.stdout)
            
            # Try to determine CUDA version
            cuda_version_line = [line for line in result.stdout.split('\n') if 'CUDA Version' in line]
            if cuda_version_line:
                cuda_version = cuda_version_line[0].split('CUDA Version:')[1].strip()
                print(f"Detected CUDA Version: {cuda_version}")
                return cuda_version
            else:
                print("Could not determine CUDA version from nvidia-smi output.")
                return "unknown"
        else:
            print("CUDA is not available or nvidia-smi cannot be executed.")
            return None
    except FileNotFoundError:
        print("nvidia-smi not found. CUDA might not be installed or available.")
        return None

def setup_environment_venv(env_path="medsam2_env"):
    """Set up the virtual environment for MedSAM2 using venv."""
    # Check if Python is available
    try:
        subprocess.run(["python3", "--version"], check=True, capture_output=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        try:
            subprocess.run(["python", "--version"], check=True, capture_output=True)
            python_cmd = "python"
        except:
            print("Python not found. Please install Python or provide the correct path.")
            return False
    else:
        python_cmd = "python3"
    
    # Check if environment already exists
    if os.path.exists(env_path):
        print(f"Environment {env_path} already exists.")
        activate_env = input(f"Use existing environment {env_path}? (y/n): ")
        if activate_env.lower() != 'y':
            return False
    else:
        # Create new environment
        if not run_command(
            f"{python_cmd} -m venv {env_path}",
            "Creating virtual environment"
        ):
            return False
    
    # Determine activation command based on platform
    if sys.platform == 'win32':
        activate_cmd = f"call {env_path}\\Scripts\\activate && "
    else:
        activate_cmd = f"source {env_path}/bin/activate && "
    
    # Install packages
    # Check CUDA version and install appropriate PyTorch version
    cuda_version = check_cuda()
    if cuda_version:
        if cuda_version.startswith("12"):
            pytorch_cmd = "pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121"
        elif cuda_version.startswith("11"):
            pytorch_cmd = "pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118"
        else:
            pytorch_cmd = "pip install torch==2.3.1 torchvision==0.18.1"
    else:
        # CPU version for systems without CUDA
        pytorch_cmd = "pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu"
    
    if not run_command(
        activate_cmd + pytorch_cmd,
        "Installing PyTorch"
    ):
        return False
    
    # Install additional dependencies
    if not run_command(
        activate_cmd + "pip install matplotlib scipy scikit-image opencv-python tqdm nibabel",
        "Installing additional packages"
    ):
        return False
    
    # Install Gradio for the UI
    if not run_command(
        activate_cmd + "pip install gradio==3.38.0",
        "Installing Gradio for UI"
    ):
        return False
    
    print(f"\nEnvironment {env_path} is ready!")
    print(f"To activate, run: source {env_path}/bin/activate")
    return True

def clone_medsam2_repo(target_dir="MedSAM2"):
    """Clone the MedSAM2 repository."""
    if os.path.exists(target_dir):
        print(f"Directory {target_dir} already exists.")
        use_existing = input(f"Use existing repository in {target_dir}? (y/n): ")
        if use_existing.lower() == 'y':
            return True
        else:
            backup_dir = f"{target_dir}_backup_{int(time.time())}"
            print(f"Moving existing directory to {backup_dir}")
            shutil.move(target_dir, backup_dir)
    
    # Clone the repository
    return run_command(
        f"git clone -b MedSAM2 https://github.com/bowang-lab/MedSAM.git {target_dir}",
        "Cloning MedSAM2 repository"
    )

def download_sam2_model(model_dir="checkpoints", model_type="tiny"):
    """Download the SAM2 base model."""
    os.makedirs(model_dir, exist_ok=True)
    
    model_types = {
        "tiny": "sam2_hiera_tiny.pt",
        "small": "sam2_hiera_small.pt",
        "base": "sam2_hiera_base_plus.pt",
        "large": "sam2_hiera_large.pt"
    }
    
    if model_type not in model_types:
        print(f"Unknown model type: {model_type}. Using tiny model.")
        model_type = "tiny"
    
    model_filename = model_types[model_type]
    model_path = os.path.join(model_dir, model_filename)
    
    if os.path.exists(model_path):
        print(f"Model file {model_path} already exists.")
        return model_path
    
    base_url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/"
    model_url = f"{base_url}{model_filename}"
    
    print(f"Downloading {model_filename} from {model_url}...")
    try:
        urllib.request.urlretrieve(model_url, model_path)
        print(f"Downloaded to {model_path}")
        return model_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        return None

def setup_medsam2_package(repo_dir="MedSAM2", env_name="medsam2_env", conda_path=None):
    """Install the MedSAM2 package from local repository."""
    if conda_path:
        conda_cmd = os.path.join(conda_path, "conda")
    else:
        conda_cmd = "conda"
    
    # Need to activate environment and set appropriate environment variables
    activate_cmd = f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {env_name} && "
    
    # Set CUDA_HOME environment variable if not already set
    cuda_home = os.environ.get("CUDA_HOME")
    if not cuda_home:
        # Try to detect CUDA location
        potential_paths = [
            "/usr/local/cuda", 
            "/usr/local/cuda-11.0",
            "/usr/local/cuda-11.1",
            "/usr/local/cuda-11.2",
            "/usr/local/cuda-11.3",
            "/usr/local/cuda-11.4",
            "/usr/local/cuda-11.5",
            "/usr/local/cuda-11.6",
            "/usr/local/cuda-11.7",
            "/usr/local/cuda-11.8",
            "/usr/local/cuda-12.0",
            "/usr/local/cuda-12.1",
            "/usr/local/cuda-12.2"
        ]
        
        for path in potential_paths:
            if os.path.exists(path) and os.path.isdir(path):
                cuda_home = path
                break
        
        if cuda_home:
            print(f"Setting CUDA_HOME to {cuda_home}")
            os.environ["CUDA_HOME"] = cuda_home
            activate_cmd += f"export CUDA_HOME={cuda_home} && "
        else:
            print("Warning: Could not detect CUDA installation path. Set CUDA_HOME manually if needed.")
    
    # Install MedSAM2 package
    return run_command(
        activate_cmd + f"cd {repo_dir} && pip install -e .",
        "Installing MedSAM2 package"
    )

def main():
    parser = argparse.ArgumentParser(description="Setup MedSAM2 environment and repository")
    parser.add_argument("--env_name", default="medsam2_env", help="Name of conda environment")
    parser.add_argument("--conda_path", default=None, help="Path to conda installation")
    parser.add_argument("--repo_dir", default="MedSAM2", help="Directory to clone repository")
    parser.add_argument("--model_dir", default="checkpoints", help="Directory to download models")
    parser.add_argument("--model_type", default="tiny", choices=["tiny", "small", "base", "large"],
                        help="SAM2 model type to download")
    
    args = parser.parse_args()
    
    # Setup steps
    print("\n=== SETTING UP MEDSAM2 FOR MPOX LESION SEGMENTATION ===\n")
    
    # 1. Set up environment
    if not setup_environment(args.env_name, args.conda_path):
        print("Failed to set up environment. Exiting.")
        return
    
    # 2. Clone repository
    if not clone_medsam2_repo(args.repo_dir):
        print("Failed to clone repository. Exiting.")
        return
    
    # 3. Download model
    model_path = download_sam2_model(args.model_dir, args.model_type)
    if not model_path:
        print("Failed to download model. Exiting.")
        return
    
    # 4. Install MedSAM2 package
    if not setup_medsam2_package(args.repo_dir, args.env_name, args.conda_path):
        print("Failed to install MedSAM2 package. Exiting.")
        return
    
    print("\n=== SETUP COMPLETED SUCCESSFULLY ===")
    print(f"MedSAM2 repository is in: {os.path.abspath(args.repo_dir)}")
    print(f"SAM2 model is in: {os.path.abspath(model_path)}")
    print(f"\nTo use MedSAM2, activate the environment:")
    print(f"conda activate {args.env_name}")
    
    # Instructions for next steps
    print("\nNext steps:")
    print("1. Prepare your Mpox dataset using mpox_data_prep.py")
    print("2. Run inference with run_medsam2_inference.py or fine-tune with run_medsam2_finetune.py")
    
if __name__ == "__main__":
    main()
