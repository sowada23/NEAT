
"""
Convert a genome's network output into a game action.

A genome produces numeric outputs, but the game expects
3 button actions such as move left, move right, or jump.
This file turns raw network output into the action format
that SlimeVolley understands.

In short:
- runs the genome forward
- converts outputs into 0/1 button presses
- returns a valid SlimeVolley action
"""


import numpy as np

def genome_to_action(genome, obs):
    """Convert a genome output vector into SlimeVolley's 3-button action."""
    outputs = genome.forward(obs)
    outputs = np.asarray(outputs, dtype=np.float32)
    action = (outputs > 0.5).astype(np.int8)
    return action.tolist()
