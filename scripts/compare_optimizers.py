"""SGD vs Muon vs AdamW on a small AirBench (airbench94 widths 64/256/256, no distillation).

Everything but the optimizer is the AirBench recipe of src/train.py (batch 1024, warmup + linear decay, Lookahead,
label smoothing 0.2, flip/translate/Cutout). Each optimizer gets a small learning-rate sweep and its best learning
rate is picked on the 5k validation images (mean over seeds).

    python -m scripts.compare_optimizers                                # 3 optimizers x 3 lrs x (10, 30) epochs x 2 seeds
    python -m scripts.compare_optimizers --epochs 20 --seeds 0 --optimizers sgd muon

Curves of each run go to logs/<name>/<run_id>/<run>/, the tables to runs.csv (every run) and best.csv.
"""
import argparse
import json
import os

import pandas as pd
import torch

from src.data import GpuLoader, load_cifar10
from src.models import AirBench, count_parameters
from src.results import new_run_dir
from src.train import AIRBENCH_HYP, evaluate, train

WIDTHS = (64, 256, 256)
LR_KEY = {"sgd": "lr", "muon": "muon_lr", "adamw": "adam_lr"}
LEARNING_RATES = {
    "sgd": [4.5, 9.0, 18.0],     # airbench units (make_optimizer), 9.0 is the AirBench default
    "muon": [0.12, 0.24, 0.48],  # conv filters only, 0.24 is airbench94_muon's
    "adamw": [1e-3, 2e-3, 4e-3],
}
ADAM_WEIGHT_DECAY = 0.05


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="63_optimizer_comparison")
    parser.add_argument("--optimizers", nargs="+", default=list(LEARNING_RATES), choices=list(LEARNING_RATES))
    parser.add_argument("--epochs", nargs="+", type=int, default=[10, 30])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    n_params, n_total = count_parameters(AirBench(widths=WIDTHS))
    assert n_total <= 5_000_000
    print(f"Small AirBench {WIDTHS}: {n_params:,} trainable parameters (< 5M)")

    data = load_cifar10("data", val_size=5000, seed=0)
    out_dir = new_run_dir(args.name)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(dict(widths=WIDTHS, n_params=n_params, gpu=torch.cuda.get_device_name(0), base_hyp=AIRBENCH_HYP,
                       learning_rates=LEARNING_RATES, adam_weight_decay=ADAM_WEIGHT_DECAY, muon_momentum=0.6,
                       epochs=args.epochs, seeds=args.seeds), f, indent=2)

    rows = []
    for epochs in args.epochs:
        for opt in args.optimizers:
            for lr in LEARNING_RATES[opt]:
                for seed in args.seeds:
                    run = f"{opt}_lr{lr:g}_{epochs}ep_seed{seed}"
                    print(run)
                    torch.manual_seed(seed)
                    hyp = dict(AIRBENCH_HYP, epochs=epochs, optimizer=opt, adam_weight_decay=ADAM_WEIGHT_DECAY)
                    hyp[LR_KEY[opt]] = lr
                    loader = GpuLoader(*data["train"], batch_size=hyp["batch_size"], train=True,
                                       translate=hyp["translate"], cutout=hyp["cutout"])
                    model = AirBench(widths=WIDTHS).cuda().to(memory_format=torch.channels_last)
                    log_dir = os.path.join(out_dir, run)
                    os.makedirs(log_dir)
                    history, _ = train(model, loader, *data["val"], hyp, log_dir=log_dir)

                    row = dict(optimizer=opt, lr=lr, epochs=epochs, seed=seed, time_s=history["time"][-1])
                    for split in ["val", "test"]:
                        row[f"{split}_acc"] = evaluate(model, *data[split])[1]
                        row[f"{split}_tta_acc"] = evaluate(model, *data[split], tta=True)[1]
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "runs.csv"), index=False)

    runs = pd.DataFrame(rows)
    means = runs.drop(columns="seed").groupby(["epochs", "optimizer", "lr"]).mean().reset_index()
    stds = runs.groupby(["epochs", "optimizer", "lr"])["test_tta_acc"].std().values
    means["test_tta_std"] = stds
    best = means.loc[means.groupby(["epochs", "optimizer"])["val_tta_acc"].idxmax()]
    best.to_csv(os.path.join(out_dir, "best.csv"), index=False)
    print(f"Saved runs.csv and best.csv to {out_dir}/")
    print(f"Parameters {n_params:,} | best learning rate per optimizer (picked on val, mean over seeds):")
    print(best.to_string(index=False, float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
