#!/bin/bash
# Same runs as submit_full_students.sh (43-45: three WRN students, all 50k images, teachers 35 + 37 + 38, 600 epochs,
# no SAM), with logit standardization in the distillation loss (Sun et al., CVPR 2024) at its recommended
# temperature 2. Compare 46-48 with 43-45. One A100 per job.
# Run from the repo root:  bash scripts/hpc/submit_full_students_logit_std.sh
set -e
TEACHERS="35_teacher_wrn28_10_sam_seed1_full 37_teacher_wrn28_10_simclr_full 38_teacher_wrn28_10_lejepa_full"
for t in $TEACHERS; do
    [ -f checkpoints/$t.pt ] || { echo "missing checkpoints/$t.pt"; exit 1; }
done

for student in "46 wrn28_w58" "47 wrn22_w68" "48 wrn16_w86"; do
    set -- $student
    sbatch scripts/hpc/train_student.sbatch ${1}_${2}_full_t35_37_38_600ep_lstd --arch $2 --epochs 600 --teacher $TEACHERS \
        --full-train --kd-logit-std --kd-temperature 2
done
squeue -u $USER
