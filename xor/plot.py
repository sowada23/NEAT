from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from xor.output_utils import latest_output_dir


def save_accuracy_svg(history: dict[str, np.ndarray], out_dir):
    out_path = out_dir / "accuracy.svg"
    generations = history["generation"]
    best_accuracy = history["best_accuracy"]
    mean_accuracy = history["mean_accuracy"]
    best_fitness = history["best_fitness"]

    plt.figure(figsize=(10, 5.8))
    plt.plot(generations, best_accuracy, color="#111111", linewidth=1.5, label="Best Accuracy")
    plt.plot(generations, mean_accuracy, color="#6c757d", linewidth=1.2, label="Mean Accuracy")
    plt.axhline(1.0, color="#2f9e44", linestyle="--", linewidth=1.0, alpha=0.8)
    plt.xlabel("Generation")
    plt.ylabel("XOR Accuracy")
    plt.ylim(-0.02, 1.05)
    plt.grid(True, alpha=0.35)

    ax2 = plt.gca().twinx()
    ax2.plot(generations, best_fitness, color="#f08c00", linewidth=1.0, alpha=0.85, label="Best Fitness")
    ax2.set_ylabel("Best Fitness")

    lines = plt.gca().lines + ax2.lines
    labels = [line.get_label() for line in lines]
    plt.legend(lines, labels, loc="lower right")
    plt.title("NEAT XOR Progress")
    plt.tight_layout()
    plt.savefig(out_path, format="svg")
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Regenerate the XOR accuracy SVG from saved history.")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = latest_output_dir() if args.out_dir is None else Path(args.out_dir).resolve()
    history = np.load(out_dir / "history.npz")
    out_path = save_accuracy_svg(history, out_dir)
    print(out_path)


if __name__ == "__main__":
    main()
