from __future__ import annotations

import argparse
import io
import pickle
from collections import defaultdict
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize

from xor.output_utils import latest_output_dir


def _node_positions(genome):
    nodes = list(genome.nodes.values())
    inputs = sorted([n for n in nodes if n.type == "input"], key=lambda n: n.id)
    hidden = sorted([n for n in nodes if n.type == "hidden"], key=lambda n: n.id)
    outputs = sorted([n for n in nodes if n.type == "output"], key=lambda n: n.id)
    enabled = [c for c in genome.connections.values() if c.enabled]

    depth = {n.id: 0 for n in inputs}
    unresolved = {n.id for n in hidden + outputs}

    for _ in range(len(nodes) + 5):
        changed = False
        for node in hidden + outputs:
            incoming = [c for c in enabled if c.out_node.id == node.id and c.in_node.id in depth]
            if incoming:
                new_depth = max(depth[c.in_node.id] + 1 for c in incoming)
                if depth.get(node.id) != new_depth:
                    depth[node.id] = new_depth
                    changed = True
                unresolved.discard(node.id)
        if not changed:
            break

    max_hidden_depth = max([depth.get(n.id, 1) for n in hidden], default=1)
    positions = {}

    def assign_vertical(node_list, x):
        if not node_list:
            return
        if len(node_list) == 1:
            positions[node_list[0].id] = (x, 0.5)
            return
        for idx, node in enumerate(node_list):
            y = 1.0 - (idx / (len(node_list) - 1))
            positions[node.id] = (x, y)

    assign_vertical(inputs, 0.0)
    hidden_by_depth = defaultdict(list)
    for node in hidden:
        hidden_by_depth[depth.get(node.id, 1)].append(node)
    for d in sorted(hidden_by_depth):
        assign_vertical(hidden_by_depth[d], d / (max_hidden_depth + 1))
    assign_vertical(outputs, 1.0)

    for node_id in unresolved:
        positions.setdefault(node_id, (0.5, 0.5))

    return positions


def render_topology_frame(genome, generation=None, accuracy=None, figsize=(8.8, 5.6)):
    positions = _node_positions(genome)
    nodes = sorted(genome.nodes.values(), key=lambda n: (n.type, n.id))
    conns = sorted(genome.connections.values(), key=lambda c: c.id)

    enabled_weights = [c.weight for c in conns if c.enabled]
    max_abs_w = max([abs(w) for w in enabled_weights], default=1.0)
    norm = Normalize(vmin=-max_abs_w, vmax=max_abs_w)
    cmap = colormaps["coolwarm"]

    fig, ax = plt.subplots(figsize=figsize, dpi=140)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    for conn in conns:
        x1, y1 = positions[conn.in_node.id]
        x2, y2 = positions[conn.out_node.id]
        touches_io = conn.in_node.type in {"input", "output"} or conn.out_node.type in {"input", "output"}

        if conn.enabled:
            color = cmap(norm(conn.weight))
            alpha = 0.85
            linewidth = 1.0 + 2.5 * min(abs(conn.weight) / max_abs_w, 1.0)
            linestyle = "-"
        else:
            color = "#bdbdbd"
            alpha = 0.45
            linewidth = 1.0
            linestyle = "--"

        ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=linewidth, linestyle=linestyle, zorder=1)

        if touches_io:
            xm = (x1 + x2) / 2.0
            ym = (y1 + y2) / 2.0
            ax.text(
                xm,
                ym,
                f"{conn.weight:.2f}",
                fontsize=6.5,
                color="#444444",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="none", alpha=0.75),
                zorder=2,
            )

    node_style = {
        "input": dict(color="#2d6cdf", size=620),
        "hidden": dict(color="#f4b400", size=680),
        "output": dict(color="#34a853", size=720),
    }

    for node in nodes:
        x, y = positions[node.id]
        style = node_style.get(node.type, dict(color="#777777", size=600))
        ax.scatter([x], [y], s=style["size"], c=style["color"], edgecolors="black", linewidths=1.2, zorder=3)

        activation = node.activation if node.activation is not None else "linear"
        label = f"id={node.id}\n{node.type}\nact={activation}\nb={node.bias:.2f}"
        ax.text(x, y, label, ha="center", va="center", fontsize=7.0, color="black", zorder=4)

    title = "NEAT XOR Topology Evolution"
    if generation is not None:
        title += f" | Generation {generation:03d}"
    if accuracy is not None:
        title += f" | accuracy={accuracy:.3f}"
    ax.set_title(title, fontsize=13, pad=10)

    ax.text(
        0.01,
        -0.08,
        "Blue=input, yellow=hidden, green=output | only connections touching input/output nodes are weight-labeled",
        transform=ax.transAxes,
        fontsize=8,
        color="#444444",
    )

    ax.set_xlim(-0.10, 1.10)
    ax.set_ylim(-0.12, 1.08)
    ax.axis("off")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return imageio.imread(buf)


def save_topology_gif(history, out_dir: Path, fps=4, sample_every=1):
    gif_path = out_dir / "topology.gif"
    sampled = history[:: max(1, sample_every)]
    frames = [render_topology_frame(genome, generation=generation, accuracy=accuracy) for generation, genome, accuracy in sampled]

    if frames:
        frames.extend([frames[-1]] * max(1, fps))
        imageio.mimsave(gif_path, frames, fps=fps)
        return gif_path
    return None


def main():
    parser = argparse.ArgumentParser(description="Regenerate the XOR topology GIF from saved history.")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--sample-every", type=int, default=1)
    args = parser.parse_args()

    out_dir = latest_output_dir() if args.out_dir is None else Path(args.out_dir).resolve()
    with open(out_dir / "topology_history.pkl", "rb") as f:
        history = pickle.load(f)
    gif_path = save_topology_gif(history, out_dir=out_dir, fps=args.fps, sample_every=args.sample_every)
    print(gif_path)


if __name__ == "__main__":
    main()
