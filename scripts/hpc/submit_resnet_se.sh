#!/bin/bash
# The three WRN students with Squeeze-and-Excitation blocks (SE-ResNets, 4.84-4.99M parameters), distilled from the
# same teacher (03) with the same recipe as 25-27 (RESNET_HYP), so 25-27 vs 40-42 isolates the effect of SE.
# One A100 per job.
# Run from the repo root:  bash scripts/hpc/submit_resnet_se.sh              (600 and 900 epochs)
#                          bash scripts/hpc/submit_resnet_se.sh 1200         (other lengths)
set -e
TEACHER=03_teacher_wrn28_10_sam
[ -f checkpoints/$TEACHER.pt ] || { echo "missing checkpoints/$TEACHER.pt: upload the teacher weights first"; exit 1; }

for epochs in ${@:-600 900}; do
    sbatch scripts/hpc/train_student.sbatch 40_wrn28_w58_se_kd_${epochs}ep --arch wrn28_w58_se --epochs $epochs --teacher $TEACHER
    sbatch scripts/hpc/train_student.sbatch 41_wrn22_w68_se_kd_${epochs}ep --arch wrn22_w68_se --epochs $epochs --teacher $TEACHER
    sbatch scripts/hpc/train_student.sbatch 42_wrn16_w86_se_kd_${epochs}ep --arch wrn16_w86_se --epochs $epochs --teacher $TEACHER
done
squeue -u $USER
