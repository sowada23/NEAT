from __future__ import annotations

from pathlib import Path

import numpy as np

from backprop_neat.genetics.genome import Genome
from backprop_neat.jax.training import TrainState, batched_forward_state, genome_to_jax
from backprop_neat.visualization.simple_png import draw_circle, save_rgb_png


def save_decision_boundary(
    genome: Genome,
    x: np.ndarray,
    y: np.ndarray,
    out_path: str | Path,
    *,
    title: str,
    grid_size: int = 220,
) -> Path:
    out_path = Path(out_path)
    margin = 0.75
    x_min, x_max = float(x[:, 0].min() - margin), float(x[:, 0].max() + margin)
    y_min, y_max = float(x[:, 1].min() - margin), float(x[:, 1].max() + margin)
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))
    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    jax_genome = genome_to_jax(genome)
    state = TrainState(jax_genome.initial_weights, jax_genome.initial_biases)
    probs = np.asarray(batched_forward_state(jax_genome, state, grid)).reshape(xx.shape)

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=140)
        contour = ax.contourf(xx, yy, probs, levels=np.linspace(0.0, 1.0, 21), cmap="RdYlBu", alpha=0.75)
        ax.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=1.2)
        labels = y.reshape(-1)
        ax.scatter(x[labels < 0.5, 0], x[labels < 0.5, 1], s=20, color="#2266aa", edgecolor="white", linewidth=0.4)
        ax.scatter(x[labels >= 0.5, 0], x[labels >= 0.5, 1], s=20, color="#cc4422", edgecolor="white", linewidth=0.4)
        fig.colorbar(contour, ax=ax, label="P(class=1)")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path
    except Exception:
        return _save_simple_boundary(out_path, x, y, probs, x_min, x_max, y_min, y_max)


def _save_simple_boundary(out_path: Path, x, y, probs, x_min, x_max, y_min, y_max) -> Path:
    p = np.clip(probs, 0.0, 1.0)
    blue = np.asarray([34, 102, 170], dtype=np.float32)
    red = np.asarray([204, 68, 34], dtype=np.float32)
    white = np.asarray([245, 245, 245], dtype=np.float32)
    image = ((1.0 - p[..., None]) * blue + p[..., None] * red) * 0.72 + white * 0.28
    image = image.astype(np.uint8)
    h, w, _ = image.shape
    labels = y.reshape(-1)
    for point, label in zip(x, labels):
        px = int((point[0] - x_min) / max(1e-6, x_max - x_min) * (w - 1))
        py = int((1.0 - (point[1] - y_min) / max(1e-6, y_max - y_min)) * (h - 1))
        color = (30, 85, 165) if label < 0.5 else (210, 60, 35)
        draw_circle(image, (px, py), 2, color)
    return save_rgb_png(out_path, image)
