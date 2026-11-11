#!/bin/bash
# Three more candidate teachers, chosen to make errors different from the supervised WRN-28-10 (03). One A100 each.
#   28: same WRN-28-10 architecture, self-supervised SimCLR pretraining (500 epochs) + supervised fine-tuning (100)
#   29: same WRN-28-10 architecture, self-supervised LeJEPA pretraining (300 epochs) + supervised fine-tuning (100)
#   30: ConvMixer-384/16 (patch 1, kernel 9), supervised, official CIFAR recipe, 300 epochs
# Run from the repo root:  bash scripts/hpc/submit_new_teachers.sh
set -e
sbatch scripts/hpc/pretrain_ssl.sbatch 28_teacher_wrn28_10_simclr --method simclr
sbatch scripts/hpc/pretrain_ssl.sbatch 29_teacher_wrn28_10_lejepa --method lejepa
sbatch scripts/hpc/train_teacher.sbatch 30_teacher_convmixer --arch convmixer
squeue -u $USER
