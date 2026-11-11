#!/bin/bash
# CCT-7/3x1 teacher with the CCT long-schedule setup (sinusoidal position embedding, AdamW lr 5e-4, wd 0.05,
# RandAugment, Mixup/CutMix, random erasing), 1500 epochs, all 50k images, without label smoothing (used as a
# distillation teacher). ~6 h on one A100.  Run from the repo root:  bash scripts/hpc/submit_cct_sine.sh
set -e
sbatch scripts/hpc/train_teacher.sbatch 56_teacher_cct7_3x1_sine_full --arch cct_sine --full-train
squeue -u $USER
