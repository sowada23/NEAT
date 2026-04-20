
"""
Visualize the structure of a NEAT genome.

This file draws the network topology of a genome:
- input nodes
- hidden nodes
- output nodes
- connections and weights

It can render one frame or combine many frames into a GIF
to show how the topology changes over generations.

In short:
- computes node positions
- draws nodes and connections
- renders topology images
- saves topology evolution GIFs
"""


import io
from collections import defaultdict

import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize


def _node_positions(genome):
    """Lay out inputs on the left, outputs on the right, and hidden nodes by depth."""
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
        for i, node in enumerate(node_list):
            y = 1.0 - (i / (len(node_list) - 1))
            positions[node.id] = (x, y)

    assign_vertical(inputs, 0.0)

    hidden_by_depth = defaultdict(list)
    for node in hidden:
        hidden_by_depth[depth.get(node.id, 1)].append(node)
    for d in sorted(hidden_by_depth):
        x = d / (max_hidden_depth + 1)
        assign_vertical(hidden_by_depth[d], x)

    assign_vertical(outputs, 1.0)

    for node_id in unresolved:
        if node_id not in positions:
            positions[node_id] = (0.5, 0.5)

    return positions


def render_genome_topology_frame(genome, generation=None, tournament=None, benchmark_score=None, figsize=(10, 6)):
    """Render one genome as an RGB frame showing nodes, activations, and edge weights."""
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

        ax.plot(
            [x1, x2],
            [y1, y2],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=1,
        )

    node_style = {
        "input": dict(color="#2d6cdf", size=540),
        "hidden": dict(color="#f4b400", size=600),
        "output": dict(color="#34a853", size=660),
    }

    for node in nodes:
        x, y = positions[node.id]
        style = node_style.get(node.type, dict(color="#777777", size=520))
        ax.scatter(
            [x],
            [y],
            s=style["size"],
            c=style["color"],
            edgecolors="black",
            linewidths=1.2,
            zorder=3,
        )

        if node.type == "input":
            label = f"id={node.id}\n{node.type}"
        else:
            activation = node.activation if node.activation is not None else "linear"
            label = f"id={node.id}\n{node.type}\nact={activation}\nb={node.bias:.2f}"

        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=7.0,
            color="black",
            zorder=4,
        )

    title = "NEAT Topology Evolution"
    if tournament is not None:
        title += f" | Tournament {int(tournament):06d}"
    elif generation is not None:
        title += f" | Generation {generation:03d}"
    if benchmark_score is not None:
        title += f" | baseline={benchmark_score:.3f}"
    ax.set_title(title, fontsize=13, pad=10)

    ax.text(
        0.01,
        -0.08,
        "Blue=input, yellow=hidden, green=output | edge color encodes sign/magnitude | dashed=disabled",
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


def save_topology_evolution_gif(genome_history, out_dir, fps=4, sample_every=1):
    """
    Save a GIF showing only the final best-genome topology.

    genome_history is a list of tuples:
        (generation, tournament, genome_copy, baseline_benchmark_score)
    or:
        (generation, genome_copy, baseline_benchmark_score)
    """
    gif_path = out_dir / "topology_evolution.gif"
    if not genome_history:
        return None

    last_entry = genome_history[-1]
    generation = None
    tournament = None
    benchmark_score = None

    if len(last_entry) == 4:
        generation, tournament, genome, benchmark_score = last_entry
    elif len(last_entry) == 3:
        generation, genome, benchmark_score = last_entry
    else:
        raise ValueError("Unexpected genome_history entry format.")

    final_frame = render_genome_topology_frame(
        genome,
        generation=generation,
        tournament=tournament,
        benchmark_score=benchmark_score,
    )
    frames = [final_frame] * max(1, fps + 1)
    imageio.mimsave(gif_path, frames, fps=fps)
    return gif_path
