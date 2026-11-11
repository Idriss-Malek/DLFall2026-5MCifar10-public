"""Continues a trained WRN-22 student with Reinforced Iterative Classification (src/ric.py). The encoder keeps its
weights, the linear classifier is replaced by the recurrent RIC head, and the model stays under 5M parameters.

    python -m scripts.train_ric --name 62_wrn22_w68_ric --start-from 62_wrn22_w68_ric_start --epochs 200
    python -m scripts.train_ric --name 62_wrn22_w68_ric --start-from 62_wrn22_w68_ric_start --epochs 200 --resume

Trains on all 50k images and only logs the CIFAR-10 test set, like the other final runs. Metrics go to
logs/<name>/<run_id>/ (written after every epoch) and weights to checkpoints/<name>.pt.
"""
import argparse
import glob
import os

import pandas as pd
import torch

from src.data import CLASSES, GpuLoader, load_cifar10, train_indices
from src.models import STUDENTS, count_parameters
from src.results import new_run_dir, save_run
from src.ric import RIC_HYP, evaluate_ric, expected_calibration_error, train_ric
from src.train import evaluate, predict


def baseline(arch, weights, images, labels, hyp):
    """The student before RIC, measured the same way."""
    model = STUDENTS[arch]().cuda().to(memory_format=torch.channels_last)
    model.load_state_dict(weights)
    probs = torch.softmax(predict(model, images), 1)
    return {"start_test_acc": evaluate(model, images, labels)[1],
            "start_test_tta_acc": evaluate(model, images, labels, tta=True)[1],
            "start_test_ece": expected_calibration_error(probs, labels.to(probs.device), hyp["ece_bins"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--start-from", required=True, help="checkpoint name of the student to continue")
    parser.add_argument("--epochs", type=int, default=RIC_HYP["epochs"], help="RIC epochs after the head warmup")
    parser.add_argument("--head-epochs", type=int, default=RIC_HYP["head_epochs"])
    parser.add_argument("--encoder-lr", type=float, default=RIC_HYP["encoder_lr"])
    parser.add_argument("--resume", action="store_true", help="continue from checkpoints/<name>_last.pt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    start = torch.load(f"checkpoints/{args.start_from}.pt", weights_only=True)
    arch = start["hyp"]["arch"]
    hyp = dict(RIC_HYP, epochs=args.epochs, head_epochs=args.head_epochs, encoder_lr=args.encoder_lr,
               arch=f"{arch}_ric", start_from=args.start_from, sample_weights=start["hyp"]["sample_weights"],
               full_train=True,
               monitored_set="test (logged only, no validation set)")

    data = load_cifar10("data", val_size=0, seed=0)
    by_index = torch.tensor(pd.read_csv(hyp["sample_weights"], index_col="index").sort_index()["weight"].values)
    weights = by_index[train_indices("data", val_size=0, seed=0)].float()
    loader = GpuLoader(*data["train"], batch_size=hyp["batch_size"], train=True, translate=hyp["translate"],
                       cutout=hyp["cutout"], sample_weights=weights)

    model = STUDENTS[hyp["arch"]]().cuda().to(memory_format=torch.channels_last)
    incompatible = model.encoder.load_state_dict(start["model"], strict=False)
    assert not incompatible.missing_keys and set(incompatible.unexpected_keys) == {"head.weight", "head.bias"}
    n_trainable, n_total = count_parameters(model)
    assert n_total <= 5_000_000
    print(f"Student {hyp['arch']}: {n_trainable:,} trainable parameters (< 5M) "
          f"| training images: {len(data['train'][1]):,}")
    print(hyp)

    final = {"seed": args.seed, **baseline(arch, start["model"], *data["test"], hyp)}
    print(f"Before RIC ({args.start_from}): test {100 * final['start_test_acc']:.2f}% "
          f"| TTA {100 * final['start_test_tta_acc']:.2f}% | ECE {final['start_test_ece']:.4f}")

    last_path = f"checkpoints/{args.name}_last.pt"
    resume = None
    if args.resume:
        resume = torch.load(last_path, weights_only=True)
        model.load_state_dict(resume["model"])
        run_dir = sorted(glob.glob(f"logs/{args.name}/*/"))[-1]
    else:
        run_dir = new_run_dir(args.name)
    history, steps = train_ric(model, loader, *data["test"], hyp, run_dir, last_path, resume)

    final.update({f"test_{k}": v for k, v in evaluate_ric(model, *data["test"], hyp).items()})
    final["test_tta_loss"], final["test_tta_acc"] = evaluate(model, *data["test"], tta=True)
    test_x, test_y = data["test"]
    test_pred = predict(model, test_x, tta=True).argmax(1).cpu()
    confusion = torch.zeros(10, 10, dtype=torch.long)
    confusion.index_put_((test_y, test_pred), torch.ones_like(test_y), accumulate=True)
    final["per_class_test_acc"] = dict(zip(CLASSES, (confusion.diag() / confusion.sum(1)).tolist()))
    final["test_confusion_matrix"] = confusion.tolist()
    save_run(args.name, hyp, n_trainable, history, steps, final, run_dir=run_dir)

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"model": model.state_dict(), "hyp": hyp}, f"checkpoints/{args.name}.pt")
    print(f"Parameters {n_trainable:,} | trained on all 50k | TEST ACC (TTA) {100 * final['test_tta_acc']:.2f}% "
          f"| ECE {final['test_ece']:.4f}")


if __name__ == "__main__":
    main()
