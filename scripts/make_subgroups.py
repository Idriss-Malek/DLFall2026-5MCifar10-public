"""Subgroups within each class and square-root sampling weights for the 50k training images.

    python -m scripts.make_subgroups --teacher 38_teacher_wrn28_10_lejepa_full --k 20 \
        --students 43_wrn28_w58_full_t35_37_38_600ep 44_wrn22_w68_full_t35_37_38_600ep

1. Features of the 50k training images from one of our own teachers (trained from scratch on CIFAR-10).
2. k-means into k subgroups within each class.
3. Sampling weight 1/sqrt(subgroup size) per image, each class kept at 10% of the samples (train_subgroups.csv,
   used by train_student.py --sample-weights).
4. Check: each CIFAR-10 test image is assigned to the nearest subgroup of its class, and the students' test accuracy
   is reported per subgroup, to see whether rare subgroups are classified worse.
Outputs go to results/subgroups_<teacher>/<run_id>/.
"""
import argparse

import pandas as pd
import torch

from src.data import CLASSES, load_cifar10, train_indices
from src.models import load_checkpoint
from src.results import save_tables
from src.subgroups import extract_features, kmeans, sqrt_weights
from src.train import predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="38_teacher_wrn28_10_lejepa_full")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--students", nargs="*", default=[])
    args = parser.parse_args()

    data = load_cifar10("data", val_size=0, seed=0)
    (train_x, train_y), (test_x, test_y) = data["train"], data["test"]
    index = train_indices("data", val_size=0, seed=0)
    teacher = load_checkpoint(args.teacher, teacher=True)
    train_f, test_f = extract_features(teacher, train_x), extract_features(teacher, test_x)

    # class collapse check: features too tight on training images would make within-class clusters meaningless
    def spread(f, y):
        return torch.stack([(1 - f[y == c] @ f[y == c].mean(0) / f[y == c].mean(0).norm()).mean()
                            for c in range(10)]).mean().item()
    print(f"within-class spread (1 - cos to class mean): train {spread(train_f, train_y.cuda()):.3f} | "
          f"test {spread(test_f, test_y.cuda()):.3f}")

    train_group, test_group = torch.empty_like(train_y), torch.empty_like(test_y)
    for c in range(10):
        centers, assignment = kmeans(train_f[train_y.cuda() == c], args.k, seed=c)
        train_group[train_y == c] = c * args.k + assignment.cpu()
        test_group[test_y == c] = c * args.k + (test_f[test_y.cuda() == c] @ centers.T).argmax(1).cpu()
    weights = sqrt_weights(train_y, train_group)
    sizes = torch.bincount(train_group, minlength=10 * args.k)

    train_table = pd.DataFrame({"index": index.numpy(), "label": train_y.numpy(), "subgroup": train_group.numpy(),
                                "subgroup_size": sizes[train_group].numpy(), "weight": weights.numpy()})
    train_table = train_table.sort_values("index").set_index("index")
    test_table = pd.DataFrame({"label": test_y.numpy(), "subgroup": test_group.numpy(),
                               "subgroup_train_size": sizes[test_group].numpy()})
    groups = pd.DataFrame({"class": [CLASSES[g // args.k] for g in range(10 * args.k)],
                           "train_size": sizes.numpy(),
                           "test_count": torch.bincount(test_group, minlength=10 * args.k).numpy()})
    groups["train_share_in_class (%)"] = 100 * groups.train_size / 5000
    groups["test_share_in_class (%)"] = 100 * groups.test_count / 1000
    groups["sampling_share_in_class (%)"] = [
        100 * weights[train_group == g].sum().item() / 0.1 for g in range(10 * args.k)]

    for name in args.students:
        correct = (predict(load_checkpoint(name), test_x, tta=True).argmax(1).cpu() == test_y).float()
        test_table[f"correct_{name}"] = correct.numpy()
        groups[f"test_acc_{name} (%)"] = [100 * correct[test_group == g].mean().item() if (test_group == g).any()
                                          else float("nan") for g in range(10 * args.k)]

    groups["size_tercile"] = pd.qcut(groups.train_size, 3, labels=["rare", "medium", "common"])
    acc_cols = [c for c in groups if c.startswith("test_acc_")]
    by_tercile = groups.groupby("size_tercile", observed=True).apply(
        lambda d: pd.Series({"subgroups": len(d), "train images": d.train_size.sum(), "test images": d.test_count.sum(),
                             **{c: (d[c] * d.test_count).sum() / d.test_count.sum() for c in acc_cols}}))
    print(f"\nsubgroup sizes: min {sizes.min().item()}, median {sizes.median().item():.0f}, max {sizes.max().item()}")
    print(f"weight ratio max/min within classes: {(weights.max() / weights.min()).item():.2f}")
    pd.set_option("display.width", 250)
    print("\nCIFAR-10 test accuracy by subgroup size (subgroups split into terciles of training size):")
    print(by_tercile.round(2).to_string())
    if acc_cols:
        print("\nSpearman correlation, subgroup training size vs test accuracy:")
        for c in acc_cols:
            print(f"  {c}: {groups[['train_size', c]].corr(method='spearman').iloc[0, 1]:.3f}")

    save_tables(f"subgroups_{args.teacher}", {"train_subgroups": train_table, "test_subgroups": test_table,
                                              "subgroups": groups, "accuracy_by_size": by_tercile,
                                              "settings": {"teacher": args.teacher, "k": args.k}})


if __name__ == "__main__":
    main()
