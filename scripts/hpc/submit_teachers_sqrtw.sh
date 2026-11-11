#!/bin/bash
# Teachers on all 50k images with the square-root subgroup sampler (same weights as the students 49-51).
#   52: supervised WRN-28-10, SAM, 200 epochs, seed 1         (compare with 35)
#   53: SimCLR backbone of 37, fine-tuned 100 epochs           (compare with 37; no new pretraining)
#   54: LeJEPA backbone of 38, fine-tuned 100 epochs           (compare with 38; no new pretraining)
# One A100 per job.  Run from the repo root:  bash scripts/hpc/submit_teachers_sqrtw.sh
set -e
WEIGHTS=results/subgroups_38_teacher_wrn28_10_lejepa_full/20261030-213616/train_subgroups.csv
for f in $WEIGHTS checkpoints/37_teacher_wrn28_10_simclr_full_pretrained.pt \
         checkpoints/38_teacher_wrn28_10_lejepa_full_pretrained.pt; do
    [ -f $f ] || { echo "missing $f"; exit 1; }
done

sbatch scripts/hpc/train_teacher.sbatch 52_teacher_wrn28_10_sam_seed1_full_sqrtw --seed 1 --full-train --sample-weights $WEIGHTS
sbatch scripts/hpc/pretrain_ssl.sbatch 53_teacher_wrn28_10_simclr_full_sqrtw --method simclr \
    --from-pretrained 37_teacher_wrn28_10_simclr_full --full-train --sample-weights $WEIGHTS
sbatch scripts/hpc/pretrain_ssl.sbatch 54_teacher_wrn28_10_lejepa_full_sqrtw --method lejepa \
    --from-pretrained 38_teacher_wrn28_10_lejepa_full --full-train --sample-weights $WEIGHTS
squeue -u $USER
