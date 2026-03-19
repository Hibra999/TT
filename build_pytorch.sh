#!/bin/bash
set -e

# Cargar conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tt_env

# Variables de entorno para CUDA 13 y RTX 5070 (CC 12.0)
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Configurar para compilar con CC 12.0
export TORCH_CUDA_ARCH_LIST="12.0"
export USE_CUDA=1
export USE_CUDNN=0
export MAX_JOBS=8

cd /tmp/pytorch

# Instalar dependencias de Python
pip install numpy pyyaml setuptools cmake cffi typing_extensions mkl mkl-include

# Instalar ninja si no está
pip install ninja

# Limpiar builds anteriores
python setup.py clean

# Compilar e instalar
python setup.py bdist_wheel

# Instalar el wheel compilado
pip install dist/*.whl

echo "PyTorch compilado e instalado exitosamente!"
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA {torch.version.cuda}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
