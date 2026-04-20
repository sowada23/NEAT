
"""
Create a baseline score plot.

This file makes a simple SVG graph showing the actual baseline-match
score recorded at each training checkpoint.

In short:
- takes score history
- draws connected points
- saves it as an SVG file
"""


import matplotlib.pyplot as plt
import numpy as np


def save_baseline_benchmark_svg(history_score, out_dir, x_values=None, x_label="Generation"):
    out_path = out_dir / "baseline_benchmark_history.svg"

    if x_values is None:
        x = np.arange(len(history_score))
    else:
        x = np.asarray(x_values, dtype=np.float32)
    score = np.asarray(history_score, dtype=np.float32)

    plt.figure(figsize=(10, 5.8))
    plt.plot(x, score, color="black", linewidth=1.2, marker="o", markersize=4.5)
    plt.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    plt.xlabel(x_label)
    plt.ylabel("Baseline Match Score")
    plt.title("NEAT Self-Play: Actual Score vs Baseline Over Tournaments")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_path, format="svg")
    plt.close()

    return out_path
