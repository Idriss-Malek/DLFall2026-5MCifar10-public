#!/bin/bash
# Two-teacher distillation (teacher logits averaged), once both second teachers from submit_second_teachers.sh are
# trained. Same student setting as 10_kd_mix_600ep (distillation + Mixup/CutMix, 600 epochs), so 10 vs 23 vs 24 isolates
# the effect of the second teacher. One A100 per job.
# Run from the repo root:  bash scripts/hpc/submit_kd_two_teachers.sh
set -e
for t in 03_teacher_wrn28_10_sam 21_teacher_wrn28_10_sam_seed1 22_teacher_cct7_3x1; do
    [ -f checkpoints/$t.pt ] || { echo "missing checkpoints/$t.pt"; exit 1; }
done

sbatch scripts/hpc/train_student.sbatch 23_kd_mix_600ep_wrn_cct --epochs 600 --mixup 0.8 --cutmix 1.0 \
    --teacher 03_teacher_wrn28_10_sam 22_teacher_cct7_3x1
sbatch scripts/hpc/train_student.sbatch 24_kd_mix_600ep_wrn_wrn --epochs 600 --mixup 0.8 --cutmix 1.0 \
    --teacher 03_teacher_wrn28_10_sam 21_teacher_wrn28_10_sam_seed1
squeue -u $USER
