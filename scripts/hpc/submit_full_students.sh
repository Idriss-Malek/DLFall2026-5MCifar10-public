#!/bin/bash
# Final-model candidates: the three WRN students trained on all 50,000 images (no validation set), distilled from
# the average of three teachers also trained on 50k: 35 (supervised WRN-28-10), 37 (SimCLR WRN-28-10) and
# 38 (LeJEPA WRN-28-10). Same student recipe as 25-27 (RESNET_HYP: SGD, KD + Mixup/CutMix, Cutout 16, EMA).
# One A100 per job; runs longer than 600 epochs get an 8 h limit.
# Run from the repo root:  bash scripts/hpc/submit_full_students.sh                 (600 epochs, no SAM)
#                          bash scripts/hpc/submit_full_students.sh 600 900         (several lengths)
#                          SAM=0.05 bash scripts/hpc/submit_full_students.sh        (with SAM, rho = 0.05)
set -e
TEACHERS="35_teacher_wrn28_10_sam_seed1_full 37_teacher_wrn28_10_simclr_full 38_teacher_wrn28_10_lejepa_full"
for t in $TEACHERS; do
    [ -f checkpoints/$t.pt ] || { echo "missing checkpoints/$t.pt"; exit 1; }
done
SAM=${SAM:-0}
SUFFIX=$([ "$SAM" = "0" ] && echo "" || echo "_sam")

for epochs in ${@:-600}; do
    time_limit=$([ $epochs -gt 600 ] && echo 08:00:00 || echo 06:00:00)
    for student in "43 wrn28_w58" "44 wrn22_w68" "45 wrn16_w86"; do
        set -- $student
        sbatch --time=$time_limit scripts/hpc/train_student.sbatch ${1}_${2}_full_t35_37_38_${epochs}ep${SUFFIX} --arch $2 \
            --epochs $epochs --teacher $TEACHERS --sam-rho $SAM --full-train
    done
done
squeue -u $USER
