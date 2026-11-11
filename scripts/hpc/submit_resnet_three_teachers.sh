#!/bin/bash
# The three WRN students distilled from the average of three teachers: WRN-28-10 (03), WRN-28-10 seed 1 (21) and the
# CCT-7/3x1 transformer (22), whose ensemble reaches 97.90% val accuracy vs 97.60% for 03 alone. Same student recipe
# as 25-27 (RESNET_HYP), at 600 and 900 epochs, so 25-27 vs 31-33 isolates the effect of the extra teachers.
# The 900-epoch jobs get an 8 h limit: three teachers make each step slower. One A100 per job.
# Run from the repo root:  bash scripts/hpc/submit_resnet_three_teachers.sh
set -e
TEACHERS="03_teacher_wrn28_10_sam 21_teacher_wrn28_10_sam_seed1 22_teacher_cct7_3x1"
for t in $TEACHERS; do
    [ -f checkpoints/$t.pt ] || { echo "missing checkpoints/$t.pt"; exit 1; }
done

for epochs in 600 900; do
    time_limit=$([ $epochs -gt 600 ] && echo 08:00:00 || echo 06:00:00)
    for student in "31 wrn28_w58" "32 wrn22_w68" "33 wrn16_w86"; do
        set -- $student
        sbatch --time=$time_limit scripts/hpc/train_student.sbatch ${1}_${2}_kd3t_${epochs}ep --arch $2 --epochs $epochs \
            --teacher $TEACHERS
    done
done
squeue -u $USER
