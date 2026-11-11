# Distilling a Diverse Teacher Ensemble into a 5M-Parameter ResNet

**Team Whatever** · Idriss Malek (im2868) · NYU Deep Learning, Fall 2026, Project 1: CIFAR-10 Efficient ResNet challenge

Our submitted models are pre-activation Wide ResNets with under 5 million trainable parameters, trained from scratch on the 50,000 CIFAR-10 training images. They learn from the labels and from the averaged outputs of four larger teachers, which we also trained from scratch on the same images. Our report explains the design choices and what we learned.

| Final model | Trainable parameters | CIFAR-10 test accuracy (TTA) | Kaggle public score |
|---|---|---|---|
| WRN-28 (run 57) | 4,805,542 | 97.69% | 91.4% |
| WRN-22 (run 58) | 4,851,866 | 97.74% | 90.1% |

## Start here: `final_model.ipynb`

```bash
pip install -r requirements.txt
jupyter lab final_model.ipynb
```

The notebook is saved with all its outputs. Run as is, it loads the submitted WRN-22 checkpoint (`checkpoints/58_….pt`), prints its parameter count and test accuracy, and plots its training log, errors, confusion matrix and calibration. It then rebuilds the tables and figures of the report from `logs/` and `results/`. Set `TRAIN = True` in its second cell to retrain the model from random weights with the exact command used for the submission (about 3.5 hours on an A100, with the four teachers from `checkpoints/`). CIFAR-10 is downloaded to `data/` on first use.

## Every model was trained from scratch: `logs/`

No pretrained weights and no external data were used. Every network, teachers included, starts from a random initialisation. The two self-supervised teachers (SimCLR and LeJEPA) were pretrained by us on the CIFAR-10 training images only.

`logs/` holds one folder per trained model, named after its run:

| File | Content |
|---|---|
| `logs/<run>/train_log.txt` | the full console output of the run: parameter count, every hyperparameter, then progress from the first epoch |
| `logs/<run>/<run_id>/epochs.csv` | per-epoch training loss and accuracy, accuracy on the monitored set, learning rate, elapsed time |
| `logs/<run>/<run_id>/steps.csv` | per-step loss and learning rate |
| `logs/<run>/<run_id>/summary.json` | hyperparameters, parameter count, GPU, training time and final metrics |
| `logs/<run>/<run_id>/pretrain_*.csv` | self-supervised pretraining curves (SimCLR and LeJEPA teachers) |

The first epochs of every log show a network learning from scratch: the final WRN-22, for example, classifies 48% of the test images correctly after its first epoch. Slurm output from NYU HPC goes to `logs/slurm/`.

Run names start with their number, which the report and the notebook use:

| Runs | Models |
|---|---|
| 01–06 | AirBench students, without and with distillation (laptop GPU) |
| 03, 21, 22 | teachers on 45k images: WRN-28-10 with SAM (two seeds), CCT |
| 07–12, 19, 20, 23, 24 | AirBench students: Mixup/CutMix, TrivialAugment, 300 to 1200 epochs, two teachers |
| 16–18, 25–27, 40–42 | Wide ResNet students on 45k images: WRN-28, WRN-22, WRN-16, 300 to 1200 epochs, Squeeze-and-Excitation |
| 34, 35, 37, 38, 55, 56 | teachers on all 50k images: SAM-1, SAM-2, SimCLR, LeJEPA, SAM-AA, CCT |
| 52–54 | teachers retrained with the subgroup sampler |
| 43–51 | students on 50k images with the 3-teacher ensemble: plain, SAM, standardised logits, subgroup sampler |
| **57–61** | **final students**: 4-teacher ensemble and subgroup sampler, 600 and 1200 epochs |
| 62 | Reinforced Iterative Classification head on the WRN-22 (stopped early) |
| 63 | optimizer comparison: SGD, Muon and AdamW on a small AirBench |

## Training every model: `scripts/`

Every training entry point is in `scripts/`. Run them from the repository root as modules:

| Script | Trains |
|---|---|
| `scripts/train_teacher.py` | supervised teachers: WRN-28-10 (SAM, or the SAM paper recipe with AutoAugment), CCT, ConvMixer |
| `scripts/pretrain_ssl.py` | self-supervised teachers: SimCLR or LeJEPA pretraining, then fine-tuning |
| `scripts/make_subgroups.py` | subgroups and sampling weights for the weighted sampler |
| `scripts/train_student.py` | all students (AirBench and Wide ResNets), with or without distillation |
| `scripts/train_ric.py` | the RIC head (run 62) |
| `scripts/compare_optimizers.py` | the optimizer comparison (run 63) |
| `scripts/hpc/` | the Slurm jobs we submitted on NYU HPC; each `submit_*.sh` lists the exact commands of a group of runs |

The final model is built in three steps. Times are for one A100.

```bash
# 1. The four teachers of the final ensemble (a few hours each)
python -m scripts.train_teacher --name 35_teacher_wrn28_10_sam_seed1_full --seed 1 --full-train
python -m scripts.pretrain_ssl  --name 38_teacher_wrn28_10_lejepa_full --method lejepa --full-train
python -m scripts.train_teacher --name 55_teacher_wrn28_10_sampaper_seed0 --recipe sam_paper --seed 0 --full-train
python -m scripts.train_teacher --name 56_teacher_cct7_3x1_sine_full --arch cct_sine --full-train

# 2. Subgroups from the LeJEPA teacher's features, and the sampling weights (a few minutes)
python -m scripts.make_subgroups --teacher 38_teacher_wrn28_10_lejepa_full --k 20

# 3. The students (about 3.5 hours each). The submitted runs used the weights in the folder below;
#    a new run of step 2 writes them to a new folder in results/subgroups_38_teacher_wrn28_10_lejepa_full/.
WEIGHTS=results/subgroups_38_teacher_wrn28_10_lejepa_full/20261030-213616/train_subgroups.csv
TEACHERS="35_teacher_wrn28_10_sam_seed1_full 38_teacher_wrn28_10_lejepa_full 55_teacher_wrn28_10_sampaper_seed0 56_teacher_cct7_3x1_sine_full"
python -m scripts.train_student --name 57_wrn28_w58_full_t35_38_55_56_600ep_sqrtw --arch wrn28_w58 --epochs 600 \
    --full-train --teacher $TEACHERS --sample-weights $WEIGHTS
python -m scripts.train_student --name 58_wrn22_w68_full_t35_38_55_56_600ep_sqrtw --arch wrn22_w68 --epochs 600 \
    --full-train --teacher $TEACHERS --sample-weights $WEIGHTS
```

Each run writes its log to `logs/<run>/` and its weights to `checkpoints/<run>.pt`. Every other run can be reproduced from the command in its `scripts/hpc/submit_*.sh` file, or from the hyperparameters in its `summary.json`. `checkpoints/` holds the weights of every model in `logs/` except the optimizer comparison (run 63). The ten WRN-28-10 teachers (146 MB each) are stored with [Git LFS](https://git-lfs.com): run `git lfs install` before cloning to download them, or clone with `GIT_LFS_SKIP_SMUDGE=1` to skip them.

## Repository layout

```
final_model.ipynb      the final model: evaluation, plots, and the report's results
src/                   data loading and augmentation, models, training loops, self-supervised learning, subgroups, the report's figures
scripts/               every training entry point and the HPC jobs
logs/                  one folder per trained model: console output, per-epoch and per-step metrics, summary
results/               analysis tables used in the report
checkpoints/           the weights of every trained model (WRN-28-10 teachers through Git LFS)
```

## Code and methods we build on

- **AirBench** (Jordan, 2024, [KellerJordan/cifar10-airbench](https://github.com/KellerJordan/cifar10-airbench)): our AirBench student and its training recipe follow the official code; the Muon optimizer follows its `airbench94_muon` version.
- **LeJEPA** (Balestriero & LeCun, 2025): the self-supervised objective follows the official minimal recipe.
- **Compact Convolutional Transformer** (Hassani et al., 2021, [SHI-Labs/Compact-Transformers](https://github.com/SHI-Labs/Compact-Transformers)): the CCT-7/3×1 teacher and its CIFAR recipe.
- **Wide ResNets** (Zagoruyko & Komodakis, 2016) and **pre-activation residual blocks** (He et al., 2016).
- **Sharpness-Aware Minimization** (Foret et al., 2021), including the paper's WRN-28-10 recipe with AutoAugment (Cubuk et al., 2019).
- **Knowledge distillation** (Hinton et al., 2015), with the teacher seeing the student's augmented input (Beyer et al., 2022).
- **SimCLR** (Chen et al., 2020), **Mixup** (Zhang et al., 2018), **CutMix** (Yun et al., 2019), **Cutout** (DeVries & Taylor, 2017), **TrivialAugment** (Müller & Hutter, 2021), **RandAugment** (Cubuk et al., 2020), **UMAP** (McInnes et al., 2018), **RIC** (Kallel et al., 2026).

Our report gives the full references.
