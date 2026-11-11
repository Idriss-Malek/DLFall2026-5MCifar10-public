#!/bin/bash
#SBATCH --job-name=ssl_teacher
#SBATCH --account=cs_gy_6923-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/slurm/%x_%j.out

# Usage, from the repo root:  sbatch scripts/hpc/pretrain_ssl.sbatch <name> <pretrain_ssl.py arguments...>
#   e.g.  sbatch scripts/hpc/pretrain_ssl.sbatch 28_teacher_wrn28_10_simclr --method simclr
NAME=${1:?give a run name}
shift
cd "$SLURM_SUBMIT_DIR"
source scripts/hpc/paths.sh
source $CONDA_DIR/bin/activate $ENV_DIR

mkdir -p logs/$NAME
nvidia-smi || true
python -m scripts.pretrain_ssl --name $NAME --workers 8 "$@" > logs/$NAME/train_log.txt 2>&1
