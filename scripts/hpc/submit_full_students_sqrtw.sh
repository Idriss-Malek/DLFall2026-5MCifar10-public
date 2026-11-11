#!/bin/bash
# Same runs as 43-45 (three WRN students, all 50k images, teachers 35 + 37 + 38, 600 epochs, no SAM), but sampling
# the training images with weight 1/sqrt(subgroup size) inside each class (subgroups: k-means, k = 20, on the
# LeJEPA teacher's features; make_subgroups.py). Compare 49-51 with 43-45. One A100 per job.
# Run from the repo root:  bash scripts/hpc/submit_full_students_sqrtw.sh
set -e
WEIGHTS=results/subgroups_38_teacher_wrn28_10_lejepa_full/20261030-213616/train_subgroups.csv
TEACHERS="35_teacher_wrn28_10_sam_seed1_full 37_teacher_wrn28_10_simclr_full 38_teacher_wrn28_10_lejepa_full"
for t in $TEACHERS; do
    [ -f checkpoints/$t.pt ] || { echo "missing checkpoints/$t.pt"; exit 1; }
done
[ -f $WEIGHTS ] || { echo "missing $WEIGHTS: git pull"; exit 1; }

for student in "49 wrn28_w58" "50 wrn22_w68" "51 wrn16_w86"; do
    set -- $student
    sbatch scripts/hpc/train_student.sbatch ${1}_${2}_full_t35_37_38_600ep_sqrtw --arch $2 --epochs 600 --teacher $TEACHERS         --full-train --sample-weights $WEIGHTS
done
squeue -u $USER
