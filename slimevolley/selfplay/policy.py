
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

ACTION_TABLE = [
    [0, 0, 0],  # NOOP
    [1, 0, 0],  # FORWARD
    [1, 0, 1],  # FORWARD_JUMP
    [0, 0, 1],  # JUMP
    [0, 1, 1],  # BACKWARD_JUMP
    [0, 1, 0],  # BACKWARD
]
def genome_to_action(genome, obs):
    outputs = np.asarray(genome.forward(obs), dtype=np.float32)
    action_idx = int(np.argmax(outputs))
    return ACTION_TABLE[action_idx]
