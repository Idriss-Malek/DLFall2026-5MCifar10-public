"""Trains a large distillation teacher on our 45k CIFAR-10 train split: a WRN-28-10 (SGD + SAM, TrivialAugment),
a CCT-7/3x1 vision transformer (AdamW, RandAugment, Mixup/CutMix) or a ConvMixer-384/16 (AdamW, RandAugment).

    python -m scripts.train_teacher --name 03_teacher_wrn28_10_sam
    python -m scripts.train_teacher --name 21_teacher_wrn28_10_sam_seed1 --seed 1
    python -m scripts.train_teacher --name 22_teacher_cct7_3x1 --arch cct
    python -m scripts.train_teacher --name 30_teacher_convmixer --arch convmixer
    python -m scripts.train_teacher --name 34_teacher_wrn28_10_sam_full --full-train   # all 50k images, no validation set
    python -m scripts.train_teacher --epochs 2 --name smoke_test       # quick check

Metrics go to logs/<name>/<run_id>/ (written after every epoch) and weights to checkpoints/<name>.pt.
Teachers have no parameter limit: only the distilled student must stay under 5M.
"""
import argparse
import os

import pandas as pd
import torch

from src.data import (AutoAugmentCifar, CpuAugmentLoader, RandAugmentCifar, TrivialAugmentCifar, load_cifar10,
                      train_indices)
from src.models import CCT, ConvMixer, WideResNet, count_parameters
from src.results import new_run_dir, save_run
from src.train import CCT_HYP, CCT_SINE_HYP, CONVMIXER_HYP, SAM_PAPER_HYP, TEACHER_HYP, evaluate, train_standard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="03_teacher_wrn28_10_sam")
    parser.add_argument("--arch", default="wrn", choices=["wrn", "cct", "cct_sine", "convmixer"])
    parser.add_argument("--recipe", default="default", choices=["default", "sam_paper"],
                        help="WRN only: sam_paper = exact WRN-28-10 recipe of the SAM paper (AutoAugment, m-sharpness)")
    parser.add_argument("--depth", type=int, default=28, help="WRN only")
    parser.add_argument("--width", type=int, default=10, help="WRN only")
    parser.add_argument("--epochs", type=int, default=None, help="default: 200 (WRN), 1000 (CCT), 300 (ConvMixer)")
    parser.add_argument("--sam-rho", type=float, default=None, help="default: 0.05 (WRN), 0 (CCT)")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--full-train", action="store_true",
                        help="train on all 50k images (no validation set; the CIFAR-10 test set is only logged)")
    parser.add_argument("--sample-weights", default=None,
                        help="train_subgroups.csv from make_subgroups.py: weighted sampler over the training images")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    if args.arch == "cct":
        hyp = dict(CCT_HYP, arch="cct")
        model, augmented = CCT(), RandAugmentCifar
    elif args.arch == "cct_sine":
        hyp = dict(CCT_SINE_HYP, arch="cct")
        model, augmented = CCT(positional="sine"), RandAugmentCifar
    elif args.arch == "convmixer":
        hyp = dict(CONVMIXER_HYP, arch="convmixer")
        model = ConvMixer(hyp["dim"], hyp["depth"], hyp["kernel_size"], hyp["patch_size"])
        augmented = RandAugmentCifar
    elif args.recipe == "sam_paper":
        hyp = dict(SAM_PAPER_HYP, arch="wrn", recipe="sam_paper", depth=args.depth, width=args.width)
        model, augmented = WideResNet(args.depth, args.width), AutoAugmentCifar
    else:
        hyp = dict(TEACHER_HYP, arch="wrn", depth=args.depth, width=args.width)
        model, augmented = WideResNet(args.depth, args.width), TrivialAugmentCifar
    for key, value in [("epochs", args.epochs), ("sam_rho", args.sam_rho)]:
        if value is not None:
            hyp[key] = value
    hyp.update(full_train=args.full_train, sample_weights=args.sample_weights,
               monitored_set="test (logged only, no validation set)" if args.full_train else "val")

    val_size = 0 if args.full_train else 5000
    raw = load_cifar10("data", val_size=val_size, normalized=False, seed=0)
    data = load_cifar10("data", val_size=val_size, seed=0)
    monitored = data["test"] if args.full_train else data["val"]
    weights = None
    if args.sample_weights:
        by_index = torch.tensor(pd.read_csv(args.sample_weights, index_col="index").sort_index()["weight"].values)
        weights = by_index[train_indices("data", val_size=val_size, seed=0)].float()
    loader = CpuAugmentLoader(augmented(*raw["train"]), None, hyp["batch_size"], cutout=hyp["cutout"],
                              workers=args.workers, sample_weights=weights)

    model = model.cuda().to(memory_format=torch.channels_last)
    n_params = count_parameters(model)[0]
    print(f"Teacher {args.arch}: {n_params:,} trainable parameters | training images: {len(data['train'][1]):,}")
    print(hyp)

    run_dir = new_run_dir(args.name)
    history, steps = train_standard(model, loader, *monitored, hyp, log_dir=run_dir)

    final = {"seed": args.seed}
    for split in ["test"] if args.full_train else ["val", "test"]:
        for tta in [False, True]:
            key = f"{split}{'_tta' if tta else ''}"
            final[f"{key}_loss"], final[f"{key}_acc"] = evaluate(model, *data[split], tta=tta)
    save_run(args.name, hyp, n_params, history, steps, final, run_dir=run_dir)

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"model": model.state_dict(), "hyp": hyp}, f"checkpoints/{args.name}.pt")
    val = "trained on all 50k" if args.full_train else f"val acc (TTA) {100 * final['val_tta_acc']:.2f}%"
    print(f"Teacher {val} | test acc (TTA) {100 * final['test_tta_acc']:.2f}%")


if __name__ == "__main__":
    main()
