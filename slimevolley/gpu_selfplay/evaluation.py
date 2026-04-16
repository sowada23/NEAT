from __future__ import annotations

import math

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "slimevolley.gpu_selfplay.evaluation requires the optional 'jax' dependency."
    ) from exc

from slimevolley.gpu_selfplay.env import batched_observations, reset_batched_env, step_batched_env
from slimevolley.gpu_selfplay.padded_policy import (
    gather_batched_genomes,
    genomes_to_batched_policy,
    policy_actions_batched,
)


def _build_pairings(population_size: int, opponents_per_genome: int, seed_base: int) -> np.ndarray:
    rng = np.random.default_rng(seed_base)
    pairings = []
    for i in range(population_size):
        candidates = [j for j in range(population_size) if j != i]
        if not candidates:
            continue
        sample_size = min(opponents_per_genome, len(candidates))
        opponents = rng.choice(candidates, size=sample_size, replace=False)
        for j in opponents:
            pairings.append((i, int(j)))
            pairings.append((int(j), i))
    if not pairings:
        return np.zeros((0, 2), dtype=np.int32)
    return np.asarray(pairings, dtype=np.int32)


def _run_match_batch(
    batched_population,
    pairings: np.ndarray,
    seed_base: int,
    max_steps: int,
):
    keys = jax.random.split(jax.random.PRNGKey(seed_base), pairings.shape[0])
    state = reset_batched_env(keys)
    right_genomes = gather_batched_genomes(batched_population, jnp.asarray(pairings[:, 0], dtype=jnp.int32))
    left_genomes = gather_batched_genomes(batched_population, jnp.asarray(pairings[:, 1], dtype=jnp.int32))
    obs_right, obs_left = batched_observations(state)

    def body_fn(carry, _):
        state, obs_right, obs_left, total_reward, total_steps = carry
        actions_right = policy_actions_batched(right_genomes, obs_right)
        actions_left = policy_actions_batched(left_genomes, obs_left)
        next_state, next_obs_right, next_obs_left, reward, _done = step_batched_env(
            state,
            actions_right,
            actions_left,
            max_steps=max_steps,
        )
        active = (~state.done).astype(jnp.float32)
        total_reward = total_reward + reward * active
        total_steps = total_steps + active
        return (
            next_state,
            next_obs_right,
            next_obs_left,
            total_reward,
            total_steps,
        ), None

    initial = (
        state,
        obs_right,
        obs_left,
        jnp.zeros((pairings.shape[0],), dtype=jnp.float32),
        jnp.zeros((pairings.shape[0],), dtype=jnp.float32),
    )
    final_carry, _ = jax.lax.scan(body_fn, initial, xs=None, length=max_steps)
    _, _, _, rewards, steps = final_carry
    return rewards, steps


_run_match_batch_jit = jax.jit(_run_match_batch, static_argnames=("max_steps",))


def evaluate_selfplay_population_gpu(
    population,
    opponents_per_genome: int,
    seed_base: int,
    max_steps: int = 3000,
    batch_size: int = 256,
):
    pairings = _build_pairings(len(population.members), opponents_per_genome, seed_base)
    total_scores = np.zeros(len(population.members), dtype=np.float32)
    total_lengths = np.zeros(len(population.members), dtype=np.float32)
    games_played = np.zeros(len(population.members), dtype=np.int32)

    if pairings.shape[0] == 0:
        return [0.0] * len(population.members), [0.0] * len(population.members)

    batched_population = genomes_to_batched_policy(population.members)

    num_batches = math.ceil(pairings.shape[0] / batch_size)
    for batch_idx in range(num_batches):
        batch = pairings[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        rewards, steps = _run_match_batch_jit(
            batched_population=batched_population,
            pairings=batch,
            seed_base=seed_base + batch_idx,
            max_steps=max_steps,
        )
        rewards = np.asarray(rewards)
        steps = np.asarray(steps)
        for local_idx, (right_idx, left_idx) in enumerate(batch):
            reward = float(rewards[local_idx])
            step_count = float(steps[local_idx])
            total_scores[right_idx] += reward
            total_scores[left_idx] -= reward
            total_lengths[right_idx] += step_count
            total_lengths[left_idx] += step_count
            games_played[right_idx] += 1
            games_played[left_idx] += 1

    avg_scores = []
    avg_lengths = []
    for idx, genome in enumerate(population.members):
        divisor = max(1, int(games_played[idx]))
        avg_score = float(total_scores[idx] / divisor)
        avg_length = float(total_lengths[idx] / divisor)
        avg_scores.append(avg_score)
        avg_lengths.append(avg_length)
        genome.fitness = avg_score + 6.0

    return avg_scores, avg_lengths
