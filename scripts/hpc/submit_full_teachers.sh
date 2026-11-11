#!/bin/bash
# All teachers retrained on all 50,000 training images (no validation set: the CIFAR-10 test set is only logged),
# with the same recipes as their 45k-split versions. One A100 per job.
# Run from the repo root:  bash scripts/hpc/submit_full_teachers.sh
set -e
sbatch scripts/hpc/train_teacher.sbatch 34_teacher_wrn28_10_sam_full --full-train
sbatch scripts/hpc/train_teacher.sbatch 35_teacher_wrn28_10_sam_seed1_full --seed 1 --full-train
sbatch scripts/hpc/train_teacher.sbatch 36_teacher_cct7_3x1_full --arch cct --full-train
sbatch scripts/hpc/pretrain_ssl.sbatch 37_teacher_wrn28_10_simclr_full --method simclr --full-train
sbatch scripts/hpc/pretrain_ssl.sbatch 38_teacher_wrn28_10_lejepa_full --method lejepa --full-train
sbatch scripts/hpc/train_teacher.sbatch 39_teacher_convmixer_full --arch convmixer --full-train
squeue -u $USER
