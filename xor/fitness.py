from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from xor.output_utils import latest_output_dir


def save_fitness_svg(history: dict[str, np.ndarray], out_dir):
    out_path = out_dir / "fitness.svg"
    generations = history["generation"]
    best_fitness = history["best_fitness"]
    mean_fitness = history["mean_fitness"]

    plt.figure(figsize=(10, 5.8))
    plt.plot(generations, best_fitness, color="#f08c00", linewidth=1.5, label="Best Fitness")
    plt.plot(generations, mean_fitness, color="#e67700", linewidth=1.2, linestyle="--", label="Mean Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.grid(True, alpha=0.35)
    plt.title("NEAT XOR Fitness")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, format="svg")
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Regenerate the XOR fitness SVG from saved history.")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = latest_output_dir() if args.out_dir is None else Path(args.out_dir).resolve()
    history = np.load(out_dir / "history.npz")
    out_path = save_fitness_svg(history, out_dir)
    print(out_path)


if __name__ == "__main__":
    main()
