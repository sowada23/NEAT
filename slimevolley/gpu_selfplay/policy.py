"""
JAX policy helpers for SlimeVolley self-play.

This mirrors slimevolley.selfplay.policy, but evaluates NEAT genomes through
the padded JAX representation used by gpu_selfplay.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from slimevolley.gpu_selfplay.padded_policy import (
    ACTION_TABLE,
    genomes_to_batched_policy,
    policy_actions_batched,
)

BASELINE_WEIGHT = jnp.asarray(
    [
        7.5719,
        4.4285,
        2.2716,
        -0.3598,
        -7.8189,
        -2.5422,
        -3.2034,
        0.3935,
        1.2202,
        -0.49,
        -0.0316,
        0.5221,
        0.7026,
        0.4179,
        -2.1689,
        1.646,
        -13.3639,
        1.5151,
        1.1175,
        -5.3561,
        5.0442,
        0.8451,
        0.3987,
        -2.9501,
        -3.7811,
        -5.8994,
        6.4167,
        2.5014,
        7.338,
        -2.9887,
        2.4586,
        13.4191,
        2.7395,
        -3.9708,
        1.6548,
        -2.7554,
        -1.5345,
        -6.4708,
        9.2426,
        -0.7392,
        0.4452,
        1.8828,
        -2.6277,
        -10.851,
        -3.2353,
        -4.4653,
        -3.1153,
        -1.3707,
        7.318,
        16.0902,
        1.4686,
        7.0391,
        1.7765,
        -1.155,
        2.6697,
        -8.8877,
        1.1958,
        -3.2839,
        -5.4425,
        1.6809,
        7.6812,
        -2.4732,
        1.738,
        0.3781,
        0.8718,
        2.5886,
        1.6911,
        1.2953,
        -9.0052,
        -4.6038,
        -6.7447,
        -2.5528,
        0.4391,
        -4.9278,
        -3.6695,
        -4.8673,
        -1.6035,
        1.5011,
        -5.6124,
        4.9747,
        1.8998,
        3.0359,
        6.2983,
        -4.8568,
        -2.1888,
        -4.1143,
        -3.9874,
        -0.0459,
        4.7134,
        2.8952,
        -9.3627,
        -4.685,
        0.3601,
        -1.3699,
        9.7294,
        11.5596,
        0.1918,
        3.0783,
        0.0329,
        -0.1362,
        -0.1188,
        -0.7579,
        0.3278,
        -0.977,
        -0.9377,
    ],
    dtype=jnp.float32,
).reshape(7, 15)

BASELINE_BIAS = jnp.asarray(
    [2.2935, -2.0353, -1.7786, 5.4567, -3.6368, 3.4996, -0.0685],
    dtype=jnp.float32,
)


def genome_to_action(genome, obs):
    batched = genomes_to_batched_policy([genome])
    obs_batch = np.asarray([obs], dtype=np.float32)
    action = policy_actions_batched(batched, obs_batch)[0]
    return np.asarray(action, dtype=np.int8).tolist()


def reset_baseline_state(batch_size: int):
    return jnp.zeros((batch_size, 7), dtype=jnp.float32)


def baseline_actions_batched(obs_batch, output_state):
    inputs = jnp.concatenate([obs_batch[:, :8], output_state], axis=-1)
    next_output = jnp.tanh(inputs @ BASELINE_WEIGHT.T + BASELINE_BIAS)
    actions = (next_output[:, :3] > 0.75).astype(jnp.int8)
    return actions, next_output
