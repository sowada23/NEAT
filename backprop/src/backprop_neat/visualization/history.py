from __future__ import annotations

from pathlib import Path

import numpy as np


def save_history_svg(history: list[dict[str, float]], out_path: str | Path, *, metric: str, title: str, ylabel: str) -> Path:
    out_path = Path(out_path)
    generations = np.asarray([row["generation"] for row in history], dtype=float)
    best_key = f"best_{metric}"
    mean_key = f"mean_{metric}"
    series = [(best_key, "#1f77b4", "Best"), (mean_key, "#d62728", "Mean")]
    values = [np.asarray([row[key] for row in history], dtype=float) for key, _, _ in series if key in history[0]]
    if not values:
        raise ValueError(f"No history values found for metric {metric!r}")
    ymin = min(float(np.min(v)) for v in values)
    ymax = max(float(np.max(v)) for v in values)
    if abs(ymax - ymin) < 1e-12:
        ymin -= 0.5
        ymax += 0.5

    width = 900
    height = 520
    left = 72
    right = 28
    top = 54
    bottom = 64
    plot_w = width - left - right
    plot_h = height - top - bottom

    def x_px(x: float) -> float:
        if generations[-1] == generations[0]:
            return left + plot_w / 2.0
        return left + (x - generations[0]) / (generations[-1] - generations[0]) * plot_w

    def y_px(y: float) -> float:
        return top + (ymax - y) / (ymax - ymin) * plot_h

    def polyline(points: np.ndarray) -> str:
        return " ".join(f"{x_px(g):.2f},{y_px(v):.2f}" for g, v in zip(generations, points))

    y_ticks = np.linspace(ymin, ymax, 5)
    x_ticks = np.linspace(generations[0], generations[-1], min(6, len(generations)))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" font-family="Arial" font-size="20" font-weight="600">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222" stroke-width="1"/>',
    ]
    for tick in y_ticks:
        y = y_px(float(tick))
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#ddd" stroke-width="1"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="12" fill="#333">{tick:.3g}</text>')
    for tick in x_ticks:
        x = x_px(float(tick))
        parts.append(f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 5}" stroke="#222" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{top + plot_h + 24}" text-anchor="middle" font-family="Arial" font-size="12" fill="#333">{tick:.0f}</text>')
    for key, color, label in series:
        if key not in history[0]:
            continue
        ys = np.asarray([row[key] for row in history], dtype=float)
        parts.append(f'<polyline points="{polyline(ys)}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        for g, value in zip(generations, ys):
            parts.append(f'<circle cx="{x_px(float(g)):.2f}" cy="{y_px(float(value)):.2f}" r="3" fill="{color}"/>')
        legend_y = 54 + 22 * (0 if label == "Best" else 1)
        parts.append(f'<line x1="{width - 150}" y1="{legend_y}" x2="{width - 120}" y2="{legend_y}" stroke="{color}" stroke-width="2.4"/>')
        parts.append(f'<text x="{width - 112}" y="{legend_y + 4}" font-family="Arial" font-size="13" fill="#333">{label}</text>')
    parts.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="14">Generation</text>')
    parts.append(f'<text x="20" y="{top + plot_h / 2:.1f}" text-anchor="middle" transform="rotate(-90 20 {top + plot_h / 2:.1f})" font-family="Arial" font-size="14">{ylabel}</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path


def save_all_history_svgs(history: list[dict[str, float]], out_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    return {
        "fitness": save_history_svg(history, out_dir / "fitness.svg", metric="fitness", title="Backprop-NEAT Fitness", ylabel="Fitness"),
        "loss": save_history_svg(history, out_dir / "loss.svg", metric="test_loss", title="Backprop-NEAT Test Loss", ylabel="Binary cross-entropy"),
        "accuracy": save_history_svg(history, out_dir / "accuracy.svg", metric="test_accuracy", title="Backprop-NEAT Test Accuracy", ylabel="Accuracy"),
    }

