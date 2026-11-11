#!/bin/bash
# Final-model candidates: the best configuration so far (07: distillation + Mixup/CutMix) retrained on all 50,000
# CIFAR-10 training images (no validation set). Both lengths are trained; which one is used is decided by the
# 45k-split comparison 07 (300 epochs) vs 10 (600 epochs) on the validation set. One A100 per job.
# Run from the repo root:  bash scripts/hpc/submit_full_train.sh
set -e
TEACHER=03_teacher_wrn28_10_sam
[ -f checkpoints/$TEACHER.pt ] || { echo "missing checkpoints/$TEACHER.pt: upload the teacher weights first"; exit 1; }

sbatch scripts/hpc/train_student.sbatch 13_kd_mix_300ep_full --epochs 300 --teacher $TEACHER --mixup 0.8 --cutmix 1.0 --full-train
sbatch scripts/hpc/train_student.sbatch 14_kd_mix_600ep_full --epochs 600 --teacher $TEACHER --mixup 0.8 --cutmix 1.0 --full-train
squeue -u $USER
