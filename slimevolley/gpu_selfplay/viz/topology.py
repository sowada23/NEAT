"""
Visualize the structure of a NEAT genome for JAX self-play.
"""

from __future__ import annotations

from collections import defaultdict

import imageio.v2 as imageio
import numpy as np


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
        for i, node in enumerate(node_list):
            positions[node.id] = (x, 1.0 - (i / (len(node_list) - 1)))

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


def _draw_line(frame, x0, y0, x1, y1, color):
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    xs = np.linspace(x0, x1, steps + 1).astype(np.int32)
    ys = np.linspace(y0, y1, steps + 1).astype(np.int32)
    valid = (xs >= 0) & (xs < frame.shape[1]) & (ys >= 0) & (ys < frame.shape[0])
    frame[ys[valid], xs[valid]] = color


def _draw_circle(frame, cx, cy, radius, color):
    y_min = max(0, cy - radius)
    y_max = min(frame.shape[0] - 1, cy + radius)
    x_min = max(0, cx - radius)
    x_max = min(frame.shape[1] - 1, cx + radius)
    yy, xx = np.ogrid[y_min : y_max + 1, x_min : x_max + 1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
    frame[y_min : y_max + 1, x_min : x_max + 1][mask] = color


def render_genome_topology_frame(genome, generation=None, tournament=None, benchmark_score=None, figsize=(10, 6)):
    width = int(figsize[0] * 140)
    height = int(figsize[1] * 140)
    frame = np.full((height, width, 3), 255, dtype=np.uint8)
    positions = _node_positions(genome)
    margin_x = int(width * 0.08)
    margin_y = int(height * 0.12)

    def point(pos):
        x, y = pos
        return (
            int(margin_x + x * (width - 2 * margin_x)),
            int(height - margin_y - y * (height - 2 * margin_y)),
        )

    for conn in sorted(genome.connections.values(), key=lambda c: c.id):
        if conn.in_node.id not in positions or conn.out_node.id not in positions:
            continue
        x0, y0 = point(positions[conn.in_node.id])
        x1, y1 = point(positions[conn.out_node.id])
        if conn.enabled:
            color = (45, 108, 223) if conn.weight >= 0 else (220, 80, 60)
        else:
            color = (190, 190, 190)
        _draw_line(frame, x0, y0, x1, y1, color)

    color_by_type = {
        "input": (45, 108, 223),
        "hidden": (244, 180, 0),
        "output": (52, 168, 83),
    }
    radius_by_type = {"input": 12, "hidden": 14, "output": 15}
    for node in sorted(genome.nodes.values(), key=lambda n: (n.type, n.id)):
        x, y = point(positions[node.id])
        _draw_circle(
            frame,
            x,
            y,
            radius_by_type.get(node.type, 12),
            color_by_type.get(node.type, (120, 120, 120)),
        )

    return frame


def save_topology_evolution_gif(genome_history, out_dir, fps=4, sample_every=1):
    gif_path = out_dir / "topology_evolution.gif"
    if not genome_history:
        return None

    last_entry = genome_history[-1]
    if len(last_entry) == 4:
        generation, tournament, genome, benchmark_score = last_entry
    elif len(last_entry) == 3:
        generation, genome, benchmark_score = last_entry
        tournament = None
    else:
        raise ValueError("Unexpected genome_history entry format.")

    frame = render_genome_topology_frame(
        genome,
        generation=generation,
        tournament=tournament,
        benchmark_score=benchmark_score,
    )
    imageio.mimsave(gif_path, [frame] * max(1, fps + 1), fps=fps)
    return gif_path
