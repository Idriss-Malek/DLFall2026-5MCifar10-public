#!/bin/bash
# Reproduction of the SAM paper's WRN-28-10 CIFAR-10 result (Foret et al., 2021, Table 1: 97.9% = 2.1% error,
# mean of 5 runs): AutoAugment + Cutout 16, batch 256, SAM rho 0.05 with m-sharpness (8 sub-batches of 32),
# lr 0.1, wd 5e-4, cosine without warmup, 200 epochs, all 50k training images. One A100 per seed.
# Run from the repo root:  bash scripts/hpc/submit_teacher_sam_paper.sh          (seed 0)
#                          bash scripts/hpc/submit_teacher_sam_paper.sh 0 1 2    (several seeds)
set -e
for seed in ${@:-0}; do
    sbatch scripts/hpc/train_teacher.sbatch 55_teacher_wrn28_10_sampaper_seed${seed} --recipe sam_paper --seed $seed --full-train
done
squeue -u $USER
