"""
Create a baseline benchmark plot for JAX self-play.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np


def _save_fallback_svg(history_mean, history_std, out_dir, x_values=None, x_label="Tournament (K)"):
    out_path = out_dir / "baseline_benchmark_history.svg"
    mean = np.asarray(history_mean, dtype=np.float32)
    std = np.asarray(history_std, dtype=np.float32)
    if x_values is None:
        x = np.arange(len(mean), dtype=np.float32)
    else:
        x = np.asarray(x_values, dtype=np.float32)

    width = 1000
    height = 580
    pad = 70
    if len(mean) == 0:
        points = ""
    else:
        y_min = float(np.min(mean - std))
        y_max = float(np.max(mean + std))
        if y_min == y_max:
            y_min -= 1.0
            y_max += 1.0
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        if x_min == x_max:
            x_min -= 1.0
            x_max += 1.0

        def sx(v):
            return pad + (float(v) - x_min) / (x_max - x_min) * (width - 2 * pad)

        def sy(v):
            return height - pad - (float(v) - y_min) / (y_max - y_min) * (height - 2 * pad)

        points = " ".join(f"{sx(xi):.2f},{sy(yi):.2f}" for xi, yi in zip(x, mean))

    out_path.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="white"/>',
                f'<text x="{width / 2}" y="32" text-anchor="middle" font-family="sans-serif" font-size="20">NEAT: Self-Play via Random Tournament Selection</text>',
                f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-family="sans-serif" font-size="14">{x_label}</text>',
                f'<text x="20" y="{height / 2}" transform="rotate(-90 20 {height / 2})" text-anchor="middle" font-family="sans-serif" font-size="14">Average Score vs Baseline Policy</text>',
                f'<polyline points="{points}" fill="none" stroke="black" stroke-width="2"/>',
                "</svg>",
            ]
        ),
        encoding="utf-8",
    )
    return out_path


def save_baseline_benchmark_svg(history_mean, history_std, out_dir, x_values=None, x_label="Tournament (K)"):
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            import matplotlib.pyplot as plt

        out_path = out_dir / "baseline_benchmark_history.svg"
        if x_values is None:
            x = np.arange(len(history_mean), dtype=np.float32)
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
        plt.title("NEAT: Self-Play via Random Tournament Selection")
        plt.grid(True, alpha=0.35)
        plt.tight_layout()
        plt.savefig(out_path, format="svg")
        plt.close()
        return out_path
    except Exception:
        return _save_fallback_svg(history_mean, history_std, out_dir, x_values=x_values, x_label=x_label)
