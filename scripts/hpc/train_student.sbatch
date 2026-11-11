#!/bin/bash
#SBATCH --job-name=student
#SBATCH --account=cs_gy_6923-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/%x_%j.out

# Usage, from the repo root:  sbatch scripts/hpc/train_student.sbatch <name> <train_student.py arguments...>
#   e.g.  sbatch scripts/hpc/train_student.sbatch 07_kd_mix_300ep --epochs 300 --teacher 03_teacher_wrn28_10_sam --mixup 0.8
NAME=${1:?give a run name}
shift
cd "$SLURM_SUBMIT_DIR"
source scripts/hpc/paths.sh
source $CONDA_DIR/bin/activate $ENV_DIR

mkdir -p logs/$NAME
nvidia-smi || true
python -m scripts.train_student --name $NAME "$@" > logs/$NAME/train_log.txt 2>&1
