"""
Run single JAX SlimeVolley episodes.
"""

from __future__ import annotations

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "slimevolley.gpu_selfplay.episodes requires the optional 'jax' dependency."
    ) from exc

from slimevolley.gpu_selfplay.env import (
    batched_observations,
    render_state_frame,
    reset_batched_env,
    step_batched_env,
)
from slimevolley.gpu_selfplay.padded_policy import genomes_to_batched_policy, policy_actions_batched
from slimevolley.gpu_selfplay.policy import baseline_actions_batched, reset_baseline_state


def run_selfplay_episode(
    genome_right,
    genome_left,
    seed,
    capture_frames=False,
    frame_skip=2,
    max_frames=900,
    max_steps=3000,
):
    right_policy = genomes_to_batched_policy([genome_right])
    left_policy = genomes_to_batched_policy([genome_left])
    state = reset_batched_env(jax.random.split(jax.random.PRNGKey(seed), 1))
    obs_right, obs_left = batched_observations(state)
    done = False
    total_reward = 0.0
    steps = 0
    frames = []

    if capture_frames:
        frames.append(render_state_frame(state))

    while not done and steps < max_steps:
        action_right = policy_actions_batched(right_policy, obs_right)
        action_left = policy_actions_batched(left_policy, obs_left)
        state, obs_right, obs_left, reward, done_batch = step_batched_env(
            state,
            action_right,
            action_left,
            max_steps=max_steps,
        )
        total_reward += float(np.asarray(reward)[0])
        steps += 1
        done = bool(np.asarray(done_batch)[0])

        if capture_frames and len(frames) < max_frames and steps % frame_skip == 0:
            frames.append(render_state_frame(state))

    return total_reward, steps, frames


def run_vs_baseline_episode(
    genome,
    seed,
    capture_frames=False,
    frame_skip=2,
    max_frames=900,
    max_steps=3000,
):
    right_policy = genomes_to_batched_policy([genome])
    baseline_state = reset_baseline_state(1)
    state = reset_batched_env(jax.random.split(jax.random.PRNGKey(seed), 1))
    obs_right, obs_left = batched_observations(state)
    done = False
    total_reward = 0.0
    steps = 0
    frames = []

    if capture_frames:
        frames.append(render_state_frame(state))

    while not done and steps < max_steps:
        action_right = policy_actions_batched(right_policy, obs_right)
        action_left, baseline_state = baseline_actions_batched(obs_left, baseline_state)
        state, obs_right, obs_left, reward, done_batch = step_batched_env(
            state,
            action_right,
            action_left,
            max_steps=max_steps,
        )
        total_reward += float(np.asarray(reward)[0])
        steps += 1
        done = bool(np.asarray(done_batch)[0])

        if capture_frames and len(frames) < max_frames and steps % frame_skip == 0:
            frames.append(render_state_frame(state))

    return total_reward, steps, frames
