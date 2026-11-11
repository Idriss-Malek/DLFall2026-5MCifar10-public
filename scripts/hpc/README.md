# Running on NYU HPC Cloud Bursting

Course rules: use **HPC Cloud Bursting** (not the Torch cluster), through **Open OnDemand** only, with the Slurm
account `cs_gy_6923-2026fa` (300 GPU hours per student). NYU VPN is needed off campus.

| Partition | Resources |
|---|---|
| `interactive` | interactive jobs |
| `n2c48m24` | CPU only |
| `g2-standard-12` / `g2-standard-24` | 1 / 2 × NVIDIA L4 |
| `c12m85-a100-1` / `c24m170-a100-2` | 1 / 2 × NVIDIA A100 40 GB |

## One-time setup

1. Open the Cloud Bursting Open OnDemand portal (link in the course announcement) and start a **Jupyter** session
   on a CPU partition (`interactive` or `n2c48m24`) with account `cs_gy_6923-2026fa`. Open a terminal from Jupyter.
2. Get the code into `/scratch/$USER`: `git clone <repo url>`, or upload a zip through the OOD file browser.
3. For distillation runs, upload the teacher weights to `checkpoints/` (e.g. `checkpoints/03_teacher_wrn28_10_sam.pt`).
4. From the repo root, build the environment (Miniforge + Python 3.11 + `requirements.txt` in `/scratch/$USER`,
   ~10 min). NYU's Singularity/overlay recipe does not work on the cloud nodes (no `--fakeroot`, no FUSE), and is not
   needed: the PyTorch wheels include CUDA and only use the node's driver.
   and download CIFAR-10:
   ```
   bash scripts/hpc/setup_env.sh
   ```

## Running experiments

Training runs as batch jobs on an A100 (`c12m85-a100-1`), one GPU per job. The three distillation experiments
(Mixup/CutMix, TrivialAugment, both; 300 epochs each) are submitted together with:
```
bash scripts/hpc/submit_kd_experiments.sh
```
Single jobs:
```
sbatch scripts/hpc/train_teacher.sbatch 21_teacher_wrn28_10_sam_seed1 --seed 1
sbatch scripts/hpc/train_teacher.sbatch 22_teacher_cct7_3x1 --arch cct
sbatch scripts/hpc/train_student.sbatch 08_kd_mix --epochs 300 --teacher 03_teacher_wrn28_10_sam --mixup 1.0 --cutmix 1.0
```

| Command | Purpose |
|---|---|
| `squeue -u $USER` | list your jobs |
| `tail -f logs/<name>/train_log.txt` | live training progress |
| `cat logs/slurm/<job>_<id>.out` | Slurm output (GPU info, errors) |
| `scancel <job id>` | stop a job |

Metrics are written to `logs/<name>/<run_id>/` after every epoch, weights to `checkpoints/<name>.pt`.
Download both back to the local repo through the OOD file browser.

**Notes.** Cloud instances whose GPU utilization stays low for 20 minutes are shut down automatically, so keep
Jupyter sessions on CPU partitions and run all GPU work with `sbatch`. Every GPU job counts against the 300-hour
budget (a WRN-28-10 teacher with SAM takes roughly 2 GPU hours on an A100).
