#!/bin/bash
# Same runs as 49-51 (three WRN students, all 50k images, 600 epochs, no SAM, square-root subgroup sampler), with
# the teacher ensemble 35 (WRN) + 38 (LeJEPA) + 55 (SAM-paper WRN) + 56 (CCT, sinusoidal) instead of 35 + 37 + 38.
# One A100 per job.  Run from the repo root:  bash scripts/hpc/submit_full_students_4t_sqrtw.sh
set -e
WEIGHTS=results/subgroups_38_teacher_wrn28_10_lejepa_full/20261030-213616/train_subgroups.csv
TEACHERS="35_teacher_wrn28_10_sam_seed1_full 38_teacher_wrn28_10_lejepa_full 55_teacher_wrn28_10_sampaper_seed0 56_teacher_cct7_3x1_sine_full"
for f in $WEIGHTS $(for t in $TEACHERS; do echo checkpoints/$t.pt; done); do
    [ -f $f ] || { echo "missing $f"; exit 1; }
done

for student in "57 wrn28_w58" "58 wrn22_w68" "59 wrn16_w86"; do
    set -- $student
    sbatch scripts/hpc/train_student.sbatch ${1}_${2}_full_t35_38_55_56_600ep_sqrtw --arch $2 --epochs 600 --teacher $TEACHERS \
        --full-train --sample-weights $WEIGHTS
done
squeue -u $USER
