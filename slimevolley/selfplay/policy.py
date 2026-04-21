
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
    outputs = np.asarray(genome.forward(obs), dtype=np.float32)
    return (outputs > 0.0).astype(np.int8).tolist()
