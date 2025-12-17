import argparse
import csv
import os
from glob import glob
from typing import Dict, List

import matplotlib.pyplot as plt


def read_loss(path: str) -> Dict[str, List[float]]:
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        cols = [h.split(":", 1)[-1] for h in header if h]
        data = {c: [] for c in cols}
        for row in reader:
            if not row:
                continue
            row = [r for r in row if r]
            if len(row) < len(cols):
                continue
            for c, v in zip(cols, row):
                try:
                    data[c].append(float(v))
                except ValueError:
                    pass
    return data


def plot_series(out_path: str, x: List[float], series: Dict[str, List[float]], title: str, logy: bool = True):
    if not series:
        return
    plt.figure()
    for label, y in series.items():
        plt.plot(x, y, label=label)
    if logy:
        plt.yscale("log")
    plt.xlabel("epoch")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def maybe_get(data: Dict[str, List[float]], name: str) -> List[float]:
    return data.get(name, [])


def plot_loss_file(loss_path: str, out_dir: str):
    data = read_loss(loss_path)
    if not data or "epoch" not in data:
        return
    epochs = data["epoch"]

    os.makedirs(out_dir, exist_ok=True)

    # train/test total loss
    series = {}
    if "train_loss" in data:
        series["train_loss"] = data["train_loss"]
    if "test_loss" in data:
        series["test_loss"] = data["test_loss"]
    plot_series(os.path.join(out_dir, "traintest.pdf"), epochs, series, "Train/Test Loss")

    # learning rate
    if "learning_rate" in data:
        plot_series(
            os.path.join(out_dir, "learning_rate.pdf"),
            epochs,
            {"learning_rate": data["learning_rate"]},
            "Learning Rate",
        )

    # weights (if present)
    weight_series = {k: data[k] for k in data if k.startswith("weight_")}
    plot_series(os.path.join(out_dir, "weights.pdf"), epochs, weight_series, "Weights")

    # per-task losses
    tasks = ["ndens", "fluxmag", "direction", "growthrate", "unphysical", "stability"]
    loss_series = {}
    for t in tasks:
        if f"{t}_train_loss" in data:
            loss_series[f"{t}_train_loss"] = data[f"{t}_train_loss"]
        if f"{t}_test_loss" in data:
            loss_series[f"{t}_test_loss"] = data[f"{t}_test_loss"]
    plot_series(os.path.join(out_dir, "losses.pdf"), epochs, loss_series, "Per-task Losses")

    # per-task max errors
    max_series = {}
    for t in tasks:
        if f"{t}_train_max" in data:
            max_series[f"{t}_train_max"] = data[f"{t}_train_max"]
        if f"{t}_test_max" in data:
            max_series[f"{t}_test_max"] = data[f"{t}_test_max"]
    plot_series(os.path.join(out_dir, "maxes.pdf"), epochs, max_series, "Per-task Max Errors")


def main():
    parser = argparse.ArgumentParser(description="Plot loss.dat files into a plots directory.")
    parser.add_argument("--models", default="model_*", help="Glob of model directories containing loss.dat")
    parser.add_argument("--output", default="plots", help="Root output directory for plots")
    args = parser.parse_args()

    model_dirs = sorted(glob(args.models))
    if not model_dirs:
        print(f"No model directories found matching {args.models}")
        return

    for mdir in model_dirs:
        loss_path = os.path.join(mdir, "loss.dat")
        if not os.path.exists(loss_path):
            continue
        out_dir = os.path.join(args.output, os.path.basename(mdir))
        print(f"Plotting {loss_path} -> {out_dir}")
        plot_loss_file(loss_path, out_dir)


if __name__ == "__main__":
    main()
