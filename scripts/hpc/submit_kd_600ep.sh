#!/bin/bash
# Same three distillation experiments as submit_kd_experiments.sh (07-09), trained for 600 instead of 300 epochs,
# to test whether the more regularized students keep improving with longer training. One A100 per job.
# Run from the repo root:  bash scripts/hpc/submit_kd_600ep.sh
set -e
TEACHER=03_teacher_wrn28_10_sam
[ -f checkpoints/$TEACHER.pt ] || { echo "missing checkpoints/$TEACHER.pt: upload the teacher weights first"; exit 1; }

sbatch scripts/hpc/train_student.sbatch 10_kd_mix_600ep    --epochs 600 --teacher $TEACHER --mixup 0.8 --cutmix 1.0
sbatch scripts/hpc/train_student.sbatch 11_kd_ta_600ep     --epochs 600 --teacher $TEACHER --trivialaugment --workers 8
sbatch scripts/hpc/train_student.sbatch 12_kd_ta_mix_600ep --epochs 600 --teacher $TEACHER --trivialaugment --workers 8 \
                                                   --mixup 0.8 --cutmix 1.0
squeue -u $USER
