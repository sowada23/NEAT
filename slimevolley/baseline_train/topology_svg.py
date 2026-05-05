from __future__ import annotations

import argparse
import html
import pickle
from collections import defaultdict
from pathlib import Path

from slimevolley.baseline_train.policy import OUTPUT_LABELS


def _node_positions(genome):
    nodes = list(genome.nodes.values())
    inputs = sorted([n for n in nodes if n.type == "input"], key=lambda n: n.id)
    hidden = sorted([n for n in nodes if n.type == "hidden"], key=lambda n: n.id)
    outputs = sorted([n for n in nodes if n.type == "output"], key=lambda n: n.id)
    enabled = [c for c in genome.connections.values() if c.enabled]

    depth = {node.id: 0 for node in inputs}
    unresolved = {node.id for node in hidden + outputs}
    for _ in range(len(nodes) + 5):
        changed = False
        for node in hidden + outputs:
            incoming = [c for c in enabled if c.out_node.id == node.id and c.in_node.id in depth]
            if incoming:
                next_depth = max(depth[c.in_node.id] + 1 for c in incoming)
                if depth.get(node.id) != next_depth:
                    depth[node.id] = next_depth
                    changed = True
                unresolved.discard(node.id)
        if not changed:
            break

    hidden_depths = [depth.get(node.id, 1) for node in hidden]
    max_hidden_depth = max(hidden_depths, default=1)
    positions = {}

    def assign_vertical(node_list, x):
        if not node_list:
            return
        if len(node_list) == 1:
            positions[node_list[0].id] = (x, 0.5)
            return
        for idx, node in enumerate(node_list):
            positions[node.id] = (x, 0.92 - idx * (0.84 / (len(node_list) - 1)))

    assign_vertical(inputs, 0.08)

    by_depth = defaultdict(list)
    for node in hidden:
        by_depth[depth.get(node.id, 1)].append(node)
    for node_depth in sorted(by_depth):
        x = 0.08 + 0.84 * (node_depth / (max_hidden_depth + 1))
        assign_vertical(sorted(by_depth[node_depth], key=lambda n: n.id), x)

    assign_vertical(outputs, 0.92)
    for node_id in unresolved:
        positions.setdefault(node_id, (0.5, 0.5))
    return positions


def _weight_color(weight: float) -> str:
    if weight >= 0:
        return "#2166ac"
    return "#b2182b"


def _scale(pos, width, height, margin):
    x, y = pos
    return margin + x * (width - 2 * margin), margin + y * (height - 2 * margin)


def save_topology_svg(
    genome,
    out_path: str | Path,
    title: str = "NEAT SlimeVolley Topology",
    baseline_score: float | None = None,
    include_disabled: bool = True,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    width = 1200
    height = 760
    margin = 90
    positions = _node_positions(genome)
    nodes = sorted(genome.nodes.values(), key=lambda n: ({"input": 0, "hidden": 1, "output": 2}.get(n.type, 3), n.id))
    conns = sorted(genome.connections.values(), key=lambda c: c.id)
    enabled_weights = [abs(c.weight) for c in conns if c.enabled]
    max_abs = max(enabled_weights, default=1.0)
    enabled_count = sum(1 for c in conns if c.enabled)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,sans-serif;fill:#1f2933}",
        ".title{font-size:26px;font-weight:700}",
        ".meta{font-size:15px;fill:#4b5563}",
        ".node-label{font-size:12px;text-anchor:middle}",
        ".small{font-size:10px;fill:#374151}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="38" text-anchor="middle" class="title">{html.escape(title)}</text>',
    ]

    meta = (
        f"nodes={len(genome.nodes)} | connections={len(conns)} | enabled={enabled_count} | "
        f"fitness={getattr(genome, 'fitness', 0.0):.4f}"
    )
    if baseline_score is not None:
        meta += f" | baseline_score={baseline_score:.4f}"
    parts.append(f'<text x="{width / 2}" y="64" text-anchor="middle" class="meta">{html.escape(meta)}</text>')

    for conn in conns:
        if not conn.enabled and not include_disabled:
            continue
        x1, y1 = _scale(positions[conn.in_node.id], width, height, margin)
        x2, y2 = _scale(positions[conn.out_node.id], width, height, margin)
        stroke = _weight_color(conn.weight) if conn.enabled else "#9aa5b1"
        opacity = 0.82 if conn.enabled else 0.25
        dash = "" if conn.enabled else ' stroke-dasharray="6 6"'
        stroke_width = 1.0 + 4.0 * min(abs(conn.weight) / max_abs, 1.0) if conn.enabled else 1.0
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{stroke_width:.2f}" opacity="{opacity:.2f}"{dash}/>'
        )

    node_fill = {
        "input": "#dbeafe",
        "hidden": "#fef3c7",
        "output": "#dcfce7",
    }
    node_stroke = {
        "input": "#2563eb",
        "hidden": "#d97706",
        "output": "#16a34a",
    }

    output_ids = sorted(n.id for n in nodes if n.type == "output")
    output_name_by_id = {node_id: OUTPUT_LABELS[idx] for idx, node_id in enumerate(output_ids[: len(OUTPUT_LABELS)])}

    for node in nodes:
        x, y = _scale(positions[node.id], width, height, margin)
        fill = node_fill.get(node.type, "#e5e7eb")
        stroke = node_stroke.get(node.type, "#4b5563")
        radius = 34 if node.type != "output" else 39
        activation = node.activation if node.activation is not None else "linear"
        name = output_name_by_id.get(node.id, node.type)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        parts.append(f'<text x="{x:.1f}" y="{y - 8:.1f}" class="node-label">id {node.id}</text>')
        parts.append(f'<text x="{x:.1f}" y="{y + 7:.1f}" class="node-label">{html.escape(name)}</text>')
        if node.type != "input":
            parts.append(f'<text x="{x:.1f}" y="{y + 22:.1f}" class="node-label small">{html.escape(activation)}</text>')

    legend_y = height - 34
    parts.extend(
        [
            f'<line x1="90" y1="{legend_y}" x2="150" y2="{legend_y}" stroke="#2166ac" stroke-width="4"/>',
            f'<text x="160" y="{legend_y + 5}" class="meta">positive weight</text>',
            f'<line x1="330" y1="{legend_y}" x2="390" y2="{legend_y}" stroke="#b2182b" stroke-width="4"/>',
            f'<text x="400" y="{legend_y + 5}" class="meta">negative weight</text>',
            f'<line x1="570" y1="{legend_y}" x2="630" y2="{legend_y}" stroke="#9aa5b1" stroke-width="2" stroke-dasharray="6 6" opacity="0.5"/>',
            f'<text x="640" y="{legend_y + 5}" class="meta">disabled connection</text>',
            "</svg>",
        ]
    )

    out_path.write_text("\n".join(parts))
    return out_path


def load_genome(path: str | Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description="Save a NEAT genome topology as SVG.")
    parser.add_argument("--genome", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="NEAT SlimeVolley Topology")
    parser.add_argument("--baseline-score", type=float, default=None)
    parser.add_argument("--hide-disabled", action="store_true")
    args = parser.parse_args()
    genome = load_genome(args.genome)
    path = save_topology_svg(
        genome,
        args.out,
        title=args.title,
        baseline_score=args.baseline_score,
        include_disabled=not args.hide_disabled,
    )
    print(path)


if __name__ == "__main__":
    main()

