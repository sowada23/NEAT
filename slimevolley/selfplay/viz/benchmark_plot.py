
"""
Create a benchmark score plot.

This file makes a simple SVG graph showing how the best genome
per generation performs against the baseline policy.

The graph helps you see whether training is improving over time.

In short:
- takes score history
- draws a line plot with uncertainty
- saves it as an SVG file
"""


import matplotlib.pyplot as plt
import numpy as np


def save_baseline_benchmark_svg(history_mean, history_std, out_dir, x_values=None, x_label="Generation"):
    out_path = out_dir / "baseline_benchmark_history.svg"

    if x_values is None:
        x = np.arange(len(history_mean))
    else:
        x = np.asarray(x_values, dtype=np.float32)
    mean = np.asarray(history_mean, dtype=np.float32)
    std = np.asarray(history_std, dtype=np.float32)

    plt.figure(figsize=(10, 5.8))
    plt.fill_between(x, mean - std, mean + std, color="#dfe4ea", alpha=0.55, linewidth=0)
    plt.plot(x, mean, color="black", linewidth=1.0)
    plt.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    plt.xlabel(x_label)
    plt.ylabel("Average Score vs Baseline Policy")
    plt.title("NEAT Self-Play: Champion Benchmark vs Baseline Policy")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_path, format="svg")
    plt.close()

    return out_path
