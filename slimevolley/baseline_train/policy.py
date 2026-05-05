from __future__ import annotations

import numpy as np

OUTPUT_LABELS = ("forward", "backward", "jump")


def genome_to_action(genome, obs, threshold: float = 0.0) -> list[int]:
    """Convert 3 NEAT outputs into SlimeVolley button actions."""
    outputs = np.asarray(genome.forward(obs), dtype=np.float32)
    if outputs.shape[0] != 3:
        raise ValueError(f"Baseline training expects 3 outputs, got {outputs.shape[0]}.")
    return (outputs > threshold).astype(np.int8).tolist()

