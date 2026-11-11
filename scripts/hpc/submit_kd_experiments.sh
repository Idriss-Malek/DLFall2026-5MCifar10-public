#!/bin/bash
# Submits the distillation experiments as separate Slurm jobs, so each one gets its own A100.
# Run from the repo root after uploading the teacher weights:  bash scripts/hpc/submit_kd_experiments.sh
set -e
TEACHER=03_teacher_wrn28_10_sam
[ -f checkpoints/$TEACHER.pt ] || { echo "missing checkpoints/$TEACHER.pt: upload the teacher weights first"; exit 1; }

sbatch scripts/hpc/train_student.sbatch 07_kd_mix_300ep    --epochs 300 --teacher $TEACHER --mixup 0.8 --cutmix 1.0
sbatch scripts/hpc/train_student.sbatch 08_kd_ta_300ep     --epochs 300 --teacher $TEACHER --trivialaugment --workers 8
sbatch scripts/hpc/train_student.sbatch 09_kd_ta_mix_300ep --epochs 300 --teacher $TEACHER --trivialaugment --workers 8 \
                                                   --mixup 0.8 --cutmix 1.0
squeue -u $USER
