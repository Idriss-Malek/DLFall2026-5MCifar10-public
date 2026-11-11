"""Trains a student under 5M parameters (AirBench or a ResNet candidate, see STUDENTS in src/models.py),
optionally distilled from a saved teacher.

    python -m scripts.train_student --name 06_kd_airbench_300ep --epochs 300 --teacher 03_teacher_wrn28_10_sam
    python -m scripts.train_student --name smoke_test --epochs 2                          # no teacher
    python -m scripts.train_student --name ... --teacher ... --trivialaugment --mixup 0.8 --cutmix 1.0
    python -m scripts.train_student --name ... --teacher ... --full-train      # final model: all 50k training images
    python -m scripts.train_student --name ... --arch wrn16_w86 --teacher ...  # ResNet student (SGD recipe, RESNET_HYP)
    python -m scripts.train_student --name ... --teacher teacher_a teacher_b   # two teachers: their logits are averaged

Metrics go to logs/<name>/<run_id>/ (written after every epoch) and weights to checkpoints/<name>.pt.
"""
import argparse
import os

import torch

import pandas as pd

from src.data import CLASSES, CpuAugmentLoader, GpuLoader, TrivialAugmentCifar, load_cifar10, train_indices
from src.models import STUDENTS, Ensemble, build_teacher, count_parameters
from src.results import new_run_dir, save_run
from src.train import AIRBENCH_HYP, KD_HYP, RESNET_HYP, evaluate, predict, train, train_standard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--arch", default="airbench", choices=list(STUDENTS))
    parser.add_argument("--epochs", type=int, default=None, help="default: 150 (AirBench), 300 (ResNets)")
    parser.add_argument("--teacher", nargs="+", default=None,
                        help="checkpoint name(s) of the teacher(s); several are averaged; omit for no KD")
    parser.add_argument("--kd-weight", type=float, default=KD_HYP["kd_weight"])
    parser.add_argument("--kd-temperature", type=float, default=KD_HYP["kd_temperature"])
    parser.add_argument("--kd-logit-std", action="store_true",
                        help="logit standardization in the KD loss (Sun et al., 2024; use with --kd-temperature 2)")
    parser.add_argument("--mixup", type=float, default=None,
                        help="Mixup alpha (0 = off; default 0 for AirBench, 0.8 for ResNets)")
    parser.add_argument("--cutmix", type=float, default=None,
                        help="CutMix alpha (0 = off; default 0 for AirBench, 1 for ResNets)")
    parser.add_argument("--trivialaugment", action="store_true",
                        help="CPU crop/flip/TrivialAugment instead of the GPU flip/translate loader")
    parser.add_argument("--workers", type=int, default=8, help="CPU workers for --trivialaugment")
    parser.add_argument("--sam-rho", type=float, default=None, help="SAM radius for ResNet students (default 0 = off)")
    parser.add_argument("--full-train", action="store_true",
                        help="train on all 50k images (no validation set; the CIFAR-10 test set is only logged)")
    parser.add_argument("--sample-weights", default=None,
                        help="train_subgroups.csv from make_subgroups.py: weighted sampler over the training images")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    hyp = dict(AIRBENCH_HYP, epochs=150) if args.arch == "airbench" else dict(RESNET_HYP)
    if args.sam_rho is not None and args.arch == "airbench":
        parser.error("--sam-rho is only available for the ResNet students")
    for key, value in [("epochs", args.epochs), ("mixup_alpha", args.mixup), ("cutmix_alpha", args.cutmix),
                       ("sam_rho", args.sam_rho)]:
        if value is not None:
            hyp[key] = value
    hyp.update(arch=args.arch, trivialaugment=args.trivialaugment, full_train=args.full_train,
               sample_weights=args.sample_weights,
               monitored_set="test (logged only, no validation set)" if args.full_train else "val")

    teacher = None
    if args.teacher:
        teachers = []
        for name in args.teacher:
            ckpt = torch.load(f"checkpoints/{name}.pt", weights_only=True)
            t = build_teacher(ckpt["hyp"]).cuda().to(memory_format=torch.channels_last)
            t.load_state_dict(ckpt["model"])
            teachers.append(t)
        teacher = teachers[0] if len(teachers) == 1 else Ensemble(teachers)
        hyp.update(teacher=args.teacher[0] if len(args.teacher) == 1 else args.teacher,
                   kd_weight=args.kd_weight, kd_temperature=args.kd_temperature, kd_logit_std=args.kd_logit_std)

    val_size = 0 if args.full_train else 5000
    data = load_cifar10("data", val_size=val_size, seed=0)
    weights = None
    if args.sample_weights:
        by_index = torch.tensor(pd.read_csv(args.sample_weights, index_col="index").sort_index()["weight"].values)
        weights = by_index[train_indices("data", val_size=val_size, seed=0)].float()
    if args.trivialaugment:
        raw = load_cifar10("data", val_size=val_size, normalized=False, seed=0)
        loader = CpuAugmentLoader(TrivialAugmentCifar(*raw["train"]), data["train"][0][:5000], hyp["batch_size"],
                                  cutout=hyp["cutout"], workers=args.workers, sample_weights=weights)
    else:
        loader = GpuLoader(*data["train"], batch_size=hyp["batch_size"], train=True,
                           translate=hyp["translate"], cutout=hyp["cutout"], sample_weights=weights)
    model = STUDENTS[args.arch]().cuda().to(memory_format=torch.channels_last)
    n_trainable, n_total = count_parameters(model)
    assert n_total <= 5_000_000
    print(f"Student {args.arch}: {n_trainable:,} trainable parameters (< 5M) "
          f"| training images: {len(data['train'][1]):,}")
    print(hyp)

    run_dir = new_run_dir(args.name)
    monitored = data["test"] if args.full_train else data["val"]
    train_fn = train if args.arch == "airbench" else train_standard
    history, steps = train_fn(model, loader, *monitored, hyp, teacher=teacher, log_dir=run_dir)

    final = {"seed": args.seed}
    for split in ["test"] if args.full_train else ["val", "test"]:
        for tta in [False, True]:
            key = f"{split}{'_tta' if tta else ''}"
            final[f"{key}_loss"], final[f"{key}_acc"] = evaluate(model, *data[split], tta=tta)
    test_x, test_y = data["test"]
    test_pred = predict(model, test_x, tta=True).argmax(1).cpu()
    confusion = torch.zeros(10, 10, dtype=torch.long)
    confusion.index_put_((test_y, test_pred), torch.ones_like(test_y), accumulate=True)
    final["per_class_test_acc"] = dict(zip(CLASSES, (confusion.diag() / confusion.sum(1)).tolist()))
    final["test_confusion_matrix"] = confusion.tolist()
    save_run(args.name, hyp, n_trainable, history, steps, final, run_dir=run_dir)

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"model": model.state_dict(), "hyp": hyp}, f"checkpoints/{args.name}.pt")
    val = f"val acc (TTA) {100 * final['val_tta_acc']:.2f}% | " if not args.full_train else "trained on all 50k | "
    print(f"Parameters {n_trainable:,} | {val}TEST ACC (TTA) {100 * final['test_tta_acc']:.2f}%")


if __name__ == "__main__":
    main()
