from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from xor.output_utils import latest_output_dir


def save_mse_svg(history: dict[str, np.ndarray], out_dir):
    out_path = out_dir / "mse.svg"
    generations = history["generation"]
    best_mse = history["best_mse"]
    mean_mse = history["mean_mse"]

    plt.figure(figsize=(10, 5.8))
    plt.plot(generations, best_mse, color="#1971c2", linewidth=1.5, label="Best MSE")
    plt.plot(generations, mean_mse, color="#4dabf7", linewidth=1.2, linestyle="--", label="Mean MSE")
    plt.xlabel("Generation")
    plt.ylabel("Mean Squared Error")
    plt.grid(True, alpha=0.35)
    plt.title("NEAT XOR MSE")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, format="svg")
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Regenerate the XOR MSE SVG from saved history.")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = latest_output_dir() if args.out_dir is None else Path(args.out_dir).resolve()
    history = np.load(out_dir / "history.npz")
    out_path = save_mse_svg(history, out_dir)
    print(out_path)


if __name__ == "__main__":
    main()
