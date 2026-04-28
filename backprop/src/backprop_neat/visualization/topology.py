from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from backprop_neat.genetics.genome import Genome
from backprop_neat.visualization.simple_png import draw_circle, draw_line, save_rgb_png


def _positions(genome: Genome) -> dict[int, tuple[float, float]]:
    inputs = sorted([n for n in genome.nodes.values() if n.kind == "input"], key=lambda n: n.id)
    hidden = sorted([n for n in genome.nodes.values() if n.kind == "hidden"], key=lambda n: n.id)
    outputs = sorted([n for n in genome.nodes.values() if n.kind == "output"], key=lambda n: n.id)
    depth = {node.id: 0 for node in inputs}
    for _ in range(len(genome.nodes) + 2):
        for conn in genome.enabled_connections():
            if conn.in_node.id in depth:
                depth[conn.out_node.id] = max(depth.get(conn.out_node.id, 0), depth[conn.in_node.id] + 1)
    max_depth = max([depth.get(node.id, 1) for node in hidden], default=1)
    positions = {}

    def place(nodes, x):
        if not nodes:
            return
        if len(nodes) == 1:
            positions[nodes[0].id] = (x, 0.5)
            return
        for idx, node in enumerate(nodes):
            positions[node.id] = (x, 1.0 - idx / (len(nodes) - 1))

    place(inputs, 0.0)
    by_depth = defaultdict(list)
    for node in hidden:
        by_depth[depth.get(node.id, 1)].append(node)
    for d, nodes in sorted(by_depth.items()):
        place(nodes, d / (max_depth + 1))
    place(outputs, 1.0)
    return positions


def save_topology(genome: Genome, out_path: str | Path, *, title: str = "Best Backprop-NEAT Topology") -> Path:
    out_path = Path(out_path)
    positions = _positions(genome)
    try:
        import matplotlib.pyplot as plt
        from matplotlib import colormaps
        from matplotlib.colors import Normalize

        weights = [conn.weight for conn in genome.enabled_connections()]
        max_abs = max([abs(w) for w in weights], default=1.0)
        norm = Normalize(vmin=-max_abs, vmax=max_abs)
        cmap = colormaps["coolwarm"]
        fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=140)

        for conn in genome.sorted_connections:
            x1, y1 = positions[conn.in_node.id]
            x2, y2 = positions[conn.out_node.id]
            if conn.enabled:
                color = cmap(norm(conn.weight))
                alpha = 0.85
                width = 1.0 + 2.5 * min(abs(conn.weight) / max_abs, 1.0)
                style = "-"
            else:
                color = "#aaaaaa"
                alpha = 0.35
                width = 0.8
                style = "--"
            ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=width, linestyle=style, zorder=1)

        node_colors = {"input": "#3273dc", "hidden": "#f2b134", "output": "#2da44e"}
        for node in genome.sorted_nodes:
            x, y = positions[node.id]
            ax.scatter([x], [y], s=700, color=node_colors[node.kind], edgecolor="black", linewidth=1.0, zorder=2)
            activation = node.activation or "linear"
            ax.text(x, y, f"{node.id}\n{activation}\nb={node.bias:.2f}", ha="center", va="center", fontsize=7, zorder=3)
        ax.set_title(title)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path
    except Exception:
        return _save_simple_topology(genome, positions, out_path)


def _save_simple_topology(genome: Genome, positions: dict[int, tuple[float, float]], out_path: Path) -> Path:
    image = np.full((520, 800, 3), 255, dtype=np.uint8)
    weights = [conn.weight for conn in genome.enabled_connections()]
    max_abs = max([abs(w) for w in weights], default=1.0)

    def to_px(pos):
        x, y = pos
        return int(70 + x * 660), int(470 - y * 420)

    for conn in genome.sorted_connections:
        p0 = to_px(positions[conn.in_node.id])
        p1 = to_px(positions[conn.out_node.id])
        if conn.enabled:
            strength = min(abs(conn.weight) / max_abs, 1.0)
            color = (190, 55, 55) if conn.weight >= 0 else (55, 95, 190)
            width = 1 + int(4 * strength)
        else:
            color = (175, 175, 175)
            width = 1
        draw_line(image, p0, p1, color, width=width)

    colors = {"input": (50, 115, 220), "hidden": (242, 177, 52), "output": (45, 164, 78)}
    for node in genome.sorted_nodes:
        draw_circle(image, to_px(positions[node.id]), 23, colors[node.kind])
        draw_circle(image, to_px(positions[node.id]), 10, (255, 255, 255))
    return save_rgb_png(out_path, image)
