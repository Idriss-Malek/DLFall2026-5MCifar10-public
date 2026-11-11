#!/bin/bash
# Python environment shared by the HPC scripts. The cloud nodes allow neither Singularity --fakeroot nor FUSE
# mounts, so instead of NYU's container recipe we use a plain Miniforge install in scratch. PyTorch wheels ship
# their own CUDA libraries and only need the node's NVIDIA driver.
CONDA_DIR=/scratch/$USER/miniforge3
ENV_DIR=/scratch/$USER/envs/dl-cifar
