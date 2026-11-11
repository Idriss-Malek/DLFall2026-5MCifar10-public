#!/bin/bash
# Two candidate second teachers for two-teacher distillation, trained on the same 45k split as the first teacher
# (03_teacher_wrn28_10_sam). One A100 per job.
#   21: the same WRN-28-10 recipe with another seed (low-risk, little diversity)
#   22: a CCT-7/3x1 vision transformer, 1000 epochs (different architecture family, more diverse errors)
# Run from the repo root:  bash scripts/hpc/submit_second_teachers.sh
set -e
sbatch scripts/hpc/train_teacher.sbatch 21_teacher_wrn28_10_sam_seed1 --seed 1
sbatch scripts/hpc/train_teacher.sbatch 22_teacher_cct7_3x1 --arch cct
squeue -u $USER
