"""Figures of our report, read directly from logs/ (training runs) and results/ (analysis tables).

Each function returns its matplotlib figure, which final_model.ipynb shows. To save them as PDF files, from the repo root:

    python -m src.figures [<folder>]        # default folder: figures/
"""
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .data import cifar10

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, SECONDARY, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
TIE = "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8, "pdf.fonttype": 42, "text.color": INK,
    "axes.labelcolor": SECONDARY, "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": SECONDARY, "ytick.color": SECONDARY, "xtick.major.size": 0, "ytick.major.size": 0,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
})


def summary(run):
    """summary.json of the latest run folder of a trained model in logs/."""
    with open(sorted(glob.glob(f"logs/{run}/*/summary.json"))[-1], encoding="utf-8") as f:
        return json.load(f)


def test_tta(run):
    return 100 * summary(run)["test_tta_acc"]


def latest(folder, table):
    """The latest version of an analysis table in results/."""
    return pd.read_csv(sorted(glob.glob(f"results/{folder}/*/{table}.csv"))[-1])


def students_by_epochs():
    """Same teacher (03), same 45k images, Mixup + CutMix: test accuracy against training length."""
    epochs = [300, 600, 900, 1200]
    series = [("AirBench", BLUE, ["07_kd_mix_300ep", "10_kd_mix_600ep", "19_kd_mix_900ep", "20_kd_mix_1200ep"]),
              ("WRN-28", ORANGE, ["16_wrn28_w58_kd"] + [f"25_wrn28_w58_kd_{e}ep" for e in epochs[1:]]),
              ("WRN-22", AQUA, ["17_wrn22_w68_kd"] + [f"26_wrn22_w68_kd_{e}ep" for e in epochs[1:]]),
              ("WRN-16", YELLOW, ["18_wrn16_w86_kd"] + [f"27_wrn16_w86_kd_{e}ep" for e in epochs[1:]])]
    fig, ax = plt.subplots(figsize=(4.4, 2.15))
    for name, color, runs in series:
        values = [test_tta(r) for r in runs]
        ax.plot(epochs, values, color=color, lw=1.5, marker="o", ms=4.5, mec="white", mew=0.8, label=name)
        ax.annotate(name, (1200, values[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=INK)
    teacher = test_tta("03_teacher_wrn28_10_sam")
    ax.axhline(teacher, color=SECONDARY, lw=0.8, ls=(0, (4, 3)))
    ax.annotate(f"teacher {teacher:.2f}", (300, teacher), xytext=(0, 3), textcoords="offset points", color=SECONDARY)
    ax.set_xticks(epochs)
    ax.set_xlim(270, 1400)
    ax.set_ylim(96.8, 97.65)
    ax.set_xlabel("training epochs")
    ax.set_ylabel("CIFAR-10 test accuracy, TTA (%)")
    ax.legend(loc="lower right", ncol=2, handlelength=1.6, columnspacing=1.2, borderaxespad=0.1)
    fig.tight_layout(pad=0.4)
    return fig


def ensembles():
    """Every teacher and teacher ensemble, split by whether it contains the CCT."""
    teachers = latest("teacher_ensembles_with_56", "teachers")
    table = latest("teacher_ensembles_with_56", "ensembles")
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(4.4, 2.15))
    for with_cct, color, label in [(False, BLUE, "without the CCT"), (True, ORANGE, "with the CCT")]:
        single = [test_tta(f"{t}_teacher_*") for t in teachers.teacher.astype(str) if (t == "56") == with_cct]
        group = table[table["with 56"] == with_cct]
        n = np.concatenate([np.ones(len(single)), group["n"].values])
        acc = np.concatenate([single, group["test acc TTA (%)"].values])
        x = n + (-0.13 if not with_cct else 0.13) + rng.uniform(-0.09, 0.09, len(n))
        ax.scatter(x, acc, s=20, color=color, edgecolor="white", linewidth=0.5, label=label, zorder=3)
    best_student = test_tta("61_wrn22_w68_full_t35_38_55_56_1200ep_sqrtw")
    ax.axhline(best_student, color=SECONDARY, lw=0.8, ls=(0, (4, 3)))
    ax.annotate(f"best 5M student {best_student:.2f}", (6.45, best_student), xytext=(0, 3),
                textcoords="offset points", ha="right", color=SECONDARY)
    ax.set_xticks(range(1, 7))
    ax.set_xlim(0.55, 6.5)
    ax.set_ylim(97.15, 98.3)
    ax.set_xlabel("teachers in the ensemble")
    ax.set_ylabel("CIFAR-10 test accuracy, TTA (%)")
    ax.legend(loc="lower right", handletextpad=0.2, borderaxespad=0.1)
    fig.tight_layout(pad=0.4)
    return fig


LADDER = [("AirBench, 37 epochs", "01_airbench_baseline"),
          ("150 epochs", "02_airbench_150ep"),
          ("+ distillation", "04_kd_airbench"),
          ("300 epochs", "06_kd_airbench_300ep"),
          ("+ Mixup or CutMix", "07_kd_mix_300ep"),
          ("600 epochs", "10_kd_mix_600ep"),
          ("WRN-28 replaces AirBench", "25_wrn28_w58_kd_600ep"),
          ("50k images, 3 teachers", "43_wrn28_w58_full_t35_37_38_600ep"),
          ("+ subgroup sampler", "49_wrn28_w58_full_t35_37_38_600ep_sqrtw"),
          ("final 4-teacher ensemble", "57_wrn28_w58_full_t35_38_55_56_600ep_sqrtw")]


def ladder_table():
    """From the first AirBench run to the final WRN-28, one change at a time: accuracy, gain and training time."""
    rows = []
    for label, run in LADDER:
        s = summary(run)
        rows.append(dict(step=label, run=run, test_tta=100 * s["test_tta_acc"], hours=s["train_time_s"] / 3600,
                         gpu="laptop" if "Laptop" in s["gpu"] else "A100"))
    table = pd.DataFrame(rows)
    table["gain"] = table.test_tta.diff()
    return table


def ladder():
    """The ladder as a waterfall. Gains of 0.1 points or less are ties (Evaluation paragraph of the report)."""
    table = ladder_table()
    fig, ax = plt.subplots(figsize=(4.8, 2.9))
    for i, row in table.iterrows():
        time = f"{row.hours * 60:.0f} min" if row.hours < 1 else f"{row.hours:.1f} h"
        ax.annotate(f"{time}, {row.gpu}", (1.04, i), xycoords=("axes fraction", "data"), va="center",
                    color=SECONDARY, annotation_clip=False)
        if i == 0:
            ax.scatter(row.test_tta, i, s=20, color=INK, zorder=3)
            ax.annotate(f"{row.test_tta:.2f}", (row.test_tta, i), xytext=(5, 0), textcoords="offset points",
                        va="center", color=INK)
            continue
        before = table.test_tta[i - 1]
        ax.plot([before, before], [i - 1, i], color=AXIS, lw=0.6, zorder=1)
        ax.barh(i, row.gain, left=before, height=0.62, color=BLUE if row.gain > 0.1 else TIE, zorder=2)
        ax.annotate(f"{row.gain:+.2f}", (max(before, row.test_tta), i), xytext=(3, 0), textcoords="offset points",
                    va="center", color=INK)
    ax.annotate("training time", (1.04, -1.1), xycoords=("axes fraction", "data"), va="center", color=INK,
                annotation_clip=False)
    ax.set_yticks(range(len(table)), table.step)
    ax.tick_params(axis="y", colors=INK)
    ax.set_ylim(len(table) - 0.5, -1.5)
    ax.set_xlim(95.4, 98.3)
    ax.set_xticks([95.5, 96, 96.5, 97, 97.5, 98])
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("CIFAR-10 test accuracy, TTA (%)")
    handles = [Line2D([], [], ls="", marker="s", ms=5, color=BLUE, label="gain above 0.1 points"),
               Line2D([], [], ls="", marker="s", ms=5, color=TIE, label="tie (0.1 points or less)")]
    ax.legend(handles=handles, loc="lower left", handletextpad=0.2, borderaxespad=0.1)
    fig.subplots_adjust(left=0.345, right=0.79, top=0.98, bottom=0.13)
    return fig


def montage(images, gap=2):
    """Images side by side with white gaps."""
    h, w, _ = images[0].shape
    strip = np.full((h, len(images) * (w + gap) - gap, 3), 255, dtype=np.uint8)
    for i, image in enumerate(images):
        strip[:, i * (w + gap):i * (w + gap) + w] = image
    return strip


def subgroups():
    """One UMAP per class of the LeJEPA teacher's features: training images in the
    rarest third of subgroups against the rest, the final WRN-28's test errors on top, and random images from the
    largest and smallest subgroup of the class with the most errors."""
    train = latest("subgroups_umap", "train_umap")
    test = latest("subgroups_umap", "test_umap")
    groups = latest("subgroups_38_teacher_wrn28_10_lejepa_full", "subgroups")
    names = groups["class"].drop_duplicates().values
    train["rare"] = groups.size_tercile.values[train.subgroup] == "rare"
    test["rare"] = groups.size_tercile.values[test.subgroup] == "rare"
    errors = test[test.student_prediction != test.label]
    print(f"test errors of the final WRN-28 in rare subgroups: {errors.rare.sum()} of {len(errors)}")
    dark, light = "#184f95", "#d3d2ca"

    fig = plt.figure(figsize=(7.0, 3.75))
    grid = fig.add_gridspec(3, 5, height_ratios=[1, 1, 0.5], hspace=0.35, wspace=0.08)
    for c, name in enumerate(names):
        ax = fig.add_subplot(grid[c // 5, c % 5])
        for rare, color in [(False, light), (True, dark)]:
            d = train[(train.label == c) & (train.rare == rare)]
            ax.scatter(d.class_x, d.class_y, s=0.4, color=color, linewidth=0, rasterized=True)
        e = errors[errors.label == c]
        ax.scatter(e.class_x, e.class_y, s=8, color=ORANGE, edgecolor="white", linewidth=0.4, zorder=3)
        ax.set_title(f"{name}: {len(e)} errors", fontsize=8, color=INK, pad=2)
        ax.set_aspect("equal", "datalim")
        ax.axis("off")
    handles = [Line2D([], [], ls="", marker="o", ms=4, color=dark, label="rarest third of subgroups"),
               Line2D([], [], ls="", marker="o", ms=4, color=light, label="other subgroups"),
               Line2D([], [], ls="", marker="o", ms=4, color=ORANGE, mec="white", mew=0.4,
                      label="test errors of the final WRN-28")]

    worst = errors.label.value_counts().idxmax()
    sizes = groups[groups["class"] == names[worst]].train_size
    data = cifar10("data", train=True).data
    rng = np.random.default_rng(0)
    strips = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.5], hspace=0.35, wspace=0.06)
    for i, (subgroup, label) in enumerate([(sizes.idxmax(), "largest"), (sizes.idxmin(), "smallest")]):
        members = train[train.subgroup == subgroup]["index"].values
        ax = fig.add_subplot(strips[2, i])
        ax.imshow(montage(data[rng.choice(members, 8, replace=False)]), interpolation="nearest")
        ax.set_title(f"{names[worst]}, {label} subgroup ({len(members)} training images)", fontsize=8, color=INK, pad=2)
        ax.axis("off")

    fig.legend(handles=handles, loc="upper center", ncol=3, handletextpad=0.2, columnspacing=1.5,
               bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.01)
    return fig


FIGURES = {"students_by_epochs": students_by_epochs, "ensembles": ensembles, "ladder": ladder, "subgroups": subgroups}


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "figures"
    os.makedirs(out, exist_ok=True)
    os.makedirs("results/accuracy_ladder", exist_ok=True)
    ladder_table().to_csv("results/accuracy_ladder/ladder.csv", index=False)
    for name, make in FIGURES.items():
        make().savefig(f"{out}/{name}.pdf", dpi=300)
