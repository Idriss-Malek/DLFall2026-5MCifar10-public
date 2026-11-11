"""Self-supervised teacher: pretrains a WRN-28-10 backbone without labels (SimCLR or LeJEPA) on our 45k training
images, then fine-tunes it with labels into a distillation teacher. Same architecture as the supervised teacher (03),
different training method, so that its errors are less correlated with it.

    python -m scripts.pretrain_ssl --name 28_teacher_wrn28_10_simclr --method simclr
    python -m scripts.pretrain_ssl --name 29_teacher_wrn28_10_lejepa --method lejepa
    python -m scripts.pretrain_ssl --name 37_teacher_wrn28_10_simclr_full --method simclr --full-train   # all 50k images
    python -m scripts.pretrain_ssl --name smoke --method lejepa --depth 10 --width 1 --epochs 1 --finetune-epochs 1

Everything goes to logs/<name>/<run_id>/: pretrain_epochs.csv / pretrain_steps.csv (self-supervised phase, with a
k-NN accuracy every --knn-every epochs), epochs.csv / steps.csv (fine-tuning) and summary.json. Weights:
checkpoints/<name>_pretrained.pt (backbone + projector after pretraining) and checkpoints/<name>.pt (teacher).
"""
import argparse
import os

import pandas as pd
import torch

from src.data import CpuAugmentLoader, TrivialAugmentCifar, load_cifar10, train_indices
from src.models import WideResNet, count_parameters
from src.results import new_run_dir, save_run
from src.selfsup import FINETUNE_HYP, LEJEPA_HYP, SIMCLR_HYP, MultiViewCifar, SSLModel, knn_accuracy, pretrain
from src.train import evaluate, train_standard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--method", required=True, choices=["simclr", "lejepa"])
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=None, help="pretraining epochs (default 500 SimCLR, 300 LeJEPA)")
    parser.add_argument("--finetune-epochs", type=int, default=FINETUNE_HYP["epochs"])
    parser.add_argument("--knn-every", type=int, default=50)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--full-train", action="store_true",
                        help="train on all 50k images (no validation set; the CIFAR-10 test set is only logged)")
    parser.add_argument("--from-pretrained", default=None,
                        help="name of an earlier run: fine-tune its saved <name>_pretrained.pt instead of pretraining")
    parser.add_argument("--sample-weights", default=None,
                        help="train_subgroups.csv from make_subgroups.py: weighted sampler for the fine-tuning")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    ssl_hyp = dict(SIMCLR_HYP if args.method == "simclr" else LEJEPA_HYP)
    if args.epochs is not None:
        ssl_hyp["epochs"] = args.epochs

    val_size = 0 if args.full_train else 5000
    raw = load_cifar10("data", val_size=val_size, normalized=False, seed=0)
    data = load_cifar10("data", val_size=val_size, seed=0)
    monitored = data["test"] if args.full_train else data["val"]
    backbone = WideResNet(args.depth, args.width)
    model = SSLModel(backbone, backbone.head.in_features, ssl_hyp["projector"]).cuda()
    model = model.to(memory_format=torch.channels_last)
    print(f"{args.method}: WRN-{args.depth}-{args.width} backbone, {count_parameters(model)[0]:,} parameters "
          f"(with projector)")
    run_dir = new_run_dir(args.name)
    knn_data = (*data["train"], *monitored)
    os.makedirs("checkpoints", exist_ok=True)
    if args.from_pretrained:
        ckpt = torch.load(f"checkpoints/{args.from_pretrained}_pretrained.pt", weights_only=True)
        ssl_hyp = dict(ckpt["hyp"], from_pretrained=args.from_pretrained)
        model.load_state_dict(ckpt["model"])
        pretrain_time = 0.0
        print(f"fine-tuning the pretrained backbone of {args.from_pretrained} (no pretraining)")
    else:
        loader = torch.utils.data.DataLoader(
            MultiViewCifar(raw["train"][0], ssl_hyp["views"]), batch_size=ssl_hyp["batch_size"], shuffle=True,
            drop_last=True, num_workers=args.workers, pin_memory=True, persistent_workers=True)
        pretrain_time = pretrain(model, loader, ssl_hyp, knn_data, run_dir, knn_every=args.knn_every)["time"][-1]
        torch.save({"model": model.state_dict(), "hyp": ssl_hyp}, f"checkpoints/{args.name}_pretrained.pt")
    print(ssl_hyp)
    knn_pretrained = knn_accuracy(backbone, *knn_data)
    print(f"k-NN accuracy of the pretrained backbone ({'test' if args.full_train else 'val'}): "
          f"{100 * knn_pretrained:.2f}%")

    hyp = dict(FINETUNE_HYP, epochs=args.finetune_epochs, arch="wrn", depth=args.depth, width=args.width,
               pretraining=ssl_hyp, full_train=args.full_train, sample_weights=args.sample_weights,
               monitored_set="test (logged only, no validation set)" if args.full_train else "val")
    weights = None
    if args.sample_weights:
        by_index = torch.tensor(pd.read_csv(args.sample_weights, index_col="index").sort_index()["weight"].values)
        weights = by_index[train_indices("data", val_size=val_size, seed=0)].float()
    finetune_loader = CpuAugmentLoader(TrivialAugmentCifar(*raw["train"]), None, hyp["batch_size"],
                                       cutout=hyp["cutout"], workers=args.workers, sample_weights=weights)
    history, steps = train_standard(backbone, finetune_loader, *monitored, hyp, log_dir=run_dir)

    final = {"seed": args.seed, f"knn_{'test' if args.full_train else 'val'}_acc_pretrained": knn_pretrained,
             "pretrain_time_s": pretrain_time}
    for split in ["test"] if args.full_train else ["val", "test"]:
        for tta in [False, True]:
            key = f"{split}{'_tta' if tta else ''}"
            final[f"{key}_loss"], final[f"{key}_acc"] = evaluate(backbone, *data[split], tta=tta)
    save_run(args.name, hyp, count_parameters(backbone)[0], history, steps, final, run_dir=run_dir)
    torch.save({"model": backbone.state_dict(), "hyp": hyp}, f"checkpoints/{args.name}.pt")
    val = "trained on all 50k" if args.full_train else f"val acc (TTA) {100 * final['val_tta_acc']:.2f}%"
    print(f"Teacher ({args.method} + fine-tuning) {val} | test acc (TTA) {100 * final['test_tta_acc']:.2f}%")


if __name__ == "__main__":
    main()
