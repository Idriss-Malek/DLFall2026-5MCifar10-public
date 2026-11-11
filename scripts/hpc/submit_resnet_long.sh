#!/bin/bash
# Longer training of the three WRN students (16, 17, 18 at 300 epochs): same recipe (RESNET_HYP: SGD, distillation
# from the WRN-28-10 teacher + Mixup/CutMix, Cutout 16, EMA), at 600, 900 and 1200 epochs. One A100 per job.
# Run from the repo root:  bash scripts/hpc/submit_resnet_long.sh              (all three lengths)
#                          bash scripts/hpc/submit_resnet_long.sh 1200         (only some lengths)
set -e
TEACHER=03_teacher_wrn28_10_sam
[ -f checkpoints/$TEACHER.pt ] || { echo "missing checkpoints/$TEACHER.pt: upload the teacher weights first"; exit 1; }

for epochs in ${@:-600 900 1200}; do
    sbatch scripts/hpc/train_student.sbatch 25_wrn28_w58_kd_${epochs}ep --arch wrn28_w58 --epochs $epochs --teacher $TEACHER
    sbatch scripts/hpc/train_student.sbatch 26_wrn22_w68_kd_${epochs}ep --arch wrn22_w68 --epochs $epochs --teacher $TEACHER
    sbatch scripts/hpc/train_student.sbatch 27_wrn16_w86_kd_${epochs}ep --arch wrn16_w86 --epochs $epochs --teacher $TEACHER
done
squeue -u $USER
