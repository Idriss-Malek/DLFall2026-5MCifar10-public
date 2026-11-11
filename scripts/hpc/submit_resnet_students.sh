#!/bin/bash
# Architecture comparison: four pre-activation ResNet students of ~4.8-5.0M parameters (depth vs width), all trained
# with the same recipe (SGD, 300 epochs, Cutout 16, EMA of the weights, distillation from the WRN-28-10 teacher
# + Mixup/CutMix; RESNET_HYP in src/train.py). The best one on the validation set is compared with AirBench (07).
# One A100 per job.
# Run from the repo root:  bash scripts/hpc/submit_resnet_students.sh
set -e
TEACHER=03_teacher_wrn28_10_sam
[ -f checkpoints/$TEACHER.pt ] || { echo "missing checkpoints/$TEACHER.pt: upload the teacher weights first"; exit 1; }

sbatch scripts/hpc/train_student.sbatch 15_resnet18_w42_kd --arch resnet18_w42 --teacher $TEACHER
sbatch scripts/hpc/train_student.sbatch 16_wrn28_w58_kd    --arch wrn28_w58    --teacher $TEACHER
sbatch scripts/hpc/train_student.sbatch 17_wrn22_w68_kd    --arch wrn22_w68    --teacher $TEACHER
sbatch scripts/hpc/train_student.sbatch 18_wrn16_w86_kd    --arch wrn16_w86    --teacher $TEACHER
squeue -u $USER
