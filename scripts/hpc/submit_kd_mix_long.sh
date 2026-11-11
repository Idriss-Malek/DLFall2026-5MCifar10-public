#!/bin/bash
# Longer training of the best setting (distillation + Mixup/CutMix, as in 07 and 10). Validation accuracy no longer
# improved from 300 to 600 epochs, but agreement with the teacher on the shifted Kaggle images did; these runs test
# whether that trend continues. One A100 per job.
# Run from the repo root:  bash scripts/hpc/submit_kd_mix_long.sh
set -e
TEACHER=03_teacher_wrn28_10_sam
[ -f checkpoints/$TEACHER.pt ] || { echo "missing checkpoints/$TEACHER.pt: upload the teacher weights first"; exit 1; }

sbatch scripts/hpc/train_student.sbatch 19_kd_mix_900ep  --epochs 900  --teacher $TEACHER --mixup 0.8 --cutmix 1.0
sbatch scripts/hpc/train_student.sbatch 20_kd_mix_1200ep --epochs 1200 --teacher $TEACHER --mixup 0.8 --cutmix 1.0
squeue -u $USER
