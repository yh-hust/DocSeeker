#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

# 1. Install core deep learning frameworks and dependencies
pip install --no-cache-dir "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" "tensordict==0.6.2" "vllm==0.8.5.post1" torchdata
pip install "transformers[hf_xet]==4.51.1" accelerate datasets peft hf-transfer "numpy<2.0.0" "pyarrow>=15.0.0" pandas ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler pytest py-spy pyext pre-commit ruff
pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"

# 2. Install pre-compiled custom operators from official online sources
# Using --no-build-isolation for flash_attn to avoid the ABI mismatch issue discussed earlier
pip install /home/ma-user/work/dataset/dataset_yh/yh/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl --no-build-isolation
#pip install flash_attn==2.7.4.post1 --no-build-isolation

# FlashInfer requires a specific index to get the pre-built wheel matching your CUDA 12.4 and PyTorch 2.6
#pip install flashinfer==0.2.5 -i https://flashinfer.ai/whl/cu126/torch2.6/
pip install /home/ma-user/work/dataset/dataset_yh/yh/flashinfer_python-0.2.5+cu126torch2.6-cp38-abi3-linux_x86_64.whl

# 3. System-level cuDNN installation (Online Network Repo)
# Assuming Ubuntu 22.04. This replaces the local .deb cache with the official NVIDIA network repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cudnn-cuda-12

# 4. Megatron-LM directly from official GitHub repository
pip install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.12.0rc3

# 5. Additional utilities
pip install opencv-python opencv-fixer
python -c "from opencv_fixer import AutoFix; AutoFix()"
pip install decord
pip install deepspeed
pip install PyMuPDF
pip install editdistance
pip install munkres