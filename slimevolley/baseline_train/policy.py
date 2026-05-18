from __future__ import annotations

import numpy as np

OUTPUT_LABELS = ("forward", "backward", "jump")


def outputs_to_action(outputs, threshold: float = 0.0, resolve_conflict: bool = True) -> list[int]:
    """Convert 3 raw outputs into SlimeVolley button actions."""
    outputs = np.asarray(outputs, dtype=np.float32)
    action = (outputs > threshold).astype(np.int8)
    if resolve_conflict and action[0] and action[1]:
        if outputs[0] > outputs[1]:
            action[1] = 0
        elif outputs[1] > outputs[0]:
            action[0] = 0
        else:
            action[0] = 0
            action[1] = 0
    return action.tolist()


def action_name(action) -> str:
    forward, backward, jump = [int(x) for x in action]
    if forward and backward:
        return "conflict_jump" if jump else "conflict"
    if forward and jump:
        return "forward_jump"
    if backward and jump:
        return "backward_jump"
    if forward:
        return "forward"
    if backward:
        return "backward"
    if jump:
        return "jump"
    return "noop"


def genome_outputs(genome, obs):
    outputs = np.asarray(genome.forward(obs), dtype=np.float32)
    if outputs.shape[0] != 3:
        raise ValueError(f"Baseline training expects 3 outputs, got {outputs.shape[0]}.")
    return outputs


def genome_to_action(genome, obs, threshold: float = 0.0, resolve_conflict: bool = True) -> list[int]:
    """Convert 3 NEAT outputs into SlimeVolley button actions."""
    return outputs_to_action(
        genome_outputs(genome, obs),
        threshold=threshold,
        resolve_conflict=resolve_conflict,
    )


def genome_action_with_stats(genome, obs, threshold: float = 0.0, resolve_conflict: bool = True):
    """Return action plus raw/resolved action labels for behavior diagnostics."""
    outputs = genome_outputs(genome, obs)
    raw_action = outputs_to_action(outputs, threshold=threshold, resolve_conflict=False)
    action = outputs_to_action(outputs, threshold=threshold, resolve_conflict=resolve_conflict)
    return action, action_name(raw_action), action_name(action)
    outputs = np.asarray(genome.forward(obs), dtype=np.float32)
    if outputs.shape[0] != 3:
        raise ValueError(f"Baseline training expects 3 outputs, got {outputs.shape[0]}.")
    return (outputs > threshold).astype(np.int8).tolist()
