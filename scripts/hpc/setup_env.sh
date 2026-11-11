#!/bin/bash
# One-time setup: Miniforge + a Python 3.11 environment in /scratch/$USER, then requirements.txt and CIFAR-10.
# Run from the repo root, in a terminal of an Open OnDemand Jupyter session:  bash scripts/hpc/setup_env.sh
set -e
source scripts/hpc/paths.sh

if [ ! -d $CONDA_DIR ]; then
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
        -O /scratch/$USER/miniforge.sh
    bash /scratch/$USER/miniforge.sh -b -p $CONDA_DIR
    rm /scratch/$USER/miniforge.sh
fi
if [ ! -d $ENV_DIR ]; then
    $CONDA_DIR/bin/conda create -y -p $ENV_DIR python=3.11
fi

source $CONDA_DIR/bin/activate $ENV_DIR
pip install -r requirements.txt
python -c "import torch, torchvision; print('torch', torch.__version__, '| torchvision', torchvision.__version__)"
python -c "import torchvision; [torchvision.datasets.CIFAR10('data', train=t, download=True) for t in (True, False)]"
echo "Environment ready: $ENV_DIR"
