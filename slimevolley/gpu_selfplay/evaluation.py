"""
JAX evaluation for SlimeVolley self-play.

This mirrors slimevolley.selfplay.evaluation, while running episode rollouts
through the batched JAX environment.
"""

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
from slimevolley.gpu_selfplay.policy import baseline_actions_batched, reset_baseline_state


def _build_index_pairings(population_size: int, opponents_per_genome: int, seed_base: int, max_tournaments=None):
    rng = np.random.default_rng(seed_base)
    pairings = []
    owners = []
    local_indices = []
    tournaments_played = 0

    for i in range(population_size):
        candidate_indices = [j for j in range(population_size) if j != i]
        if not candidate_indices:
            continue
        sample_size = min(opponents_per_genome, len(candidate_indices))
        opponents = rng.choice(candidate_indices, size=sample_size, replace=False)

        for local_idx, j in enumerate(opponents):
            if max_tournaments is not None and tournaments_played >= max_tournaments:
                break
            pairings.append((i, int(j)))
            owners.append(i)
            local_indices.append(local_idx)
            tournaments_played += 1

        if max_tournaments is not None and tournaments_played >= max_tournaments:
            break

    return (
        np.asarray(pairings, dtype=np.int32),
        np.asarray(owners, dtype=np.int32),
        np.asarray(local_indices, dtype=np.int32),
    )


def _run_match_batch(
    batched_right,
    batched_left,
    seed_base: int,
    seeds: np.ndarray,
    max_steps: int,
):
    keys = jax.vmap(lambda s: jax.random.PRNGKey(s))(jnp.asarray(seeds, dtype=jnp.uint32))
    state = reset_batched_env(keys)
    obs_right, obs_left = batched_observations(state)

    def body_fn(carry, _):
        state, obs_right, obs_left, total_reward, total_steps = carry
        actions_right = policy_actions_batched(batched_right, obs_right)
        actions_left = policy_actions_batched(batched_left, obs_left)
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

    batch_size = seeds.shape[0]
    initial = (
        state,
        obs_right,
        obs_left,
        jnp.zeros((batch_size,), dtype=jnp.float32),
        jnp.zeros((batch_size,), dtype=jnp.float32),
    )
    final_carry, _ = jax.lax.scan(body_fn, initial, xs=None, length=max_steps)
    _, _, _, rewards, steps = final_carry
    return rewards, steps


def _run_baseline_batch(
    batched_right,
    seed_base: int,
    seeds: np.ndarray,
    max_steps: int,
):
    keys = jax.vmap(lambda s: jax.random.PRNGKey(s))(jnp.asarray(seeds, dtype=jnp.uint32))
    state = reset_batched_env(keys)
    obs_right, obs_left = batched_observations(state)
    baseline_state = reset_baseline_state(seeds.shape[0])

    def body_fn(carry, _):
        state, obs_right, obs_left, baseline_state, total_reward, total_steps = carry
        actions_right = policy_actions_batched(batched_right, obs_right)
        actions_left, next_baseline_state = baseline_actions_batched(obs_left, baseline_state)
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
            next_baseline_state,
            total_reward,
            total_steps,
        ), None

    batch_size = seeds.shape[0]
    initial = (
        state,
        obs_right,
        obs_left,
        baseline_state,
        jnp.zeros((batch_size,), dtype=jnp.float32),
        jnp.zeros((batch_size,), dtype=jnp.float32),
    )
    final_carry, _ = jax.lax.scan(body_fn, initial, xs=None, length=max_steps)
    _, _, _, _, rewards, steps = final_carry
    return rewards, steps


_run_match_batch_jit = jax.jit(_run_match_batch, static_argnames=("max_steps",))
_run_baseline_batch_jit = jax.jit(_run_baseline_batch, static_argnames=("max_steps",))


def _evaluate_pairings_from_indices(population, pairings, owners, local_indices, seed_base, max_steps, batch_size):
    n = len(population.members)
    total_scores = np.zeros(n, dtype=np.float32)
    total_lengths = np.zeros(n, dtype=np.float32)
    games_played = np.zeros(n, dtype=np.int32)

    if pairings.shape[0] == 0:
        return total_scores, total_lengths, games_played

    batched_population = genomes_to_batched_policy(population.members)

    num_batches = math.ceil(pairings.shape[0] / batch_size)
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = (batch_idx + 1) * batch_size
        batch = pairings[start:end]
        batch_owners = owners[start:end]
        batch_local_indices = local_indices[start:end]

        right_first = gather_batched_genomes(
            batched_population,
            jnp.asarray(batch[:, 0], dtype=jnp.int32),
        )
        left_first = gather_batched_genomes(
            batched_population,
            jnp.asarray(batch[:, 1], dtype=jnp.int32),
        )
        first_seeds = seed_base + batch_owners * 1000 + batch_local_indices * 10
        first_rewards, first_steps = _run_match_batch_jit(
            right_first,
            left_first,
            seed_base,
            first_seeds.astype(np.uint32),
            max_steps,
        )

        right_second = gather_batched_genomes(
            batched_population,
            jnp.asarray(batch[:, 1], dtype=jnp.int32),
        )
        left_second = gather_batched_genomes(
            batched_population,
            jnp.asarray(batch[:, 0], dtype=jnp.int32),
        )
        second_seeds = first_seeds + 1
        second_rewards, second_steps = _run_match_batch_jit(
            right_second,
            left_second,
            seed_base + 1,
            second_seeds.astype(np.uint32),
            max_steps,
        )

        first_rewards = np.asarray(first_rewards)
        first_steps = np.asarray(first_steps)
        second_rewards = np.asarray(second_rewards)
        second_steps = np.asarray(second_steps)

        for local_batch_idx, owner in enumerate(batch_owners):
            total_scores[owner] += float(first_rewards[local_batch_idx] - second_rewards[local_batch_idx])
            total_lengths[owner] += float(first_steps[local_batch_idx] + second_steps[local_batch_idx])
            games_played[owner] += 2

    return total_scores, total_lengths, games_played


def _evaluate_genome_vs_baseline_batch(genomes, seeds, max_steps, batch_size):
    if not genomes:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    scores = []
    lengths = []
    num_batches = math.ceil(len(genomes) / batch_size)

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = (batch_idx + 1) * batch_size
        batched = genomes_to_batched_policy(genomes[start:end])
        batch_seeds = np.asarray(seeds[start:end], dtype=np.uint32)
        rewards, steps = _run_baseline_batch_jit(
            batched,
            int(batch_seeds[0]) if len(batch_seeds) else 0,
            batch_seeds,
            max_steps,
        )
        scores.extend(np.asarray(rewards, dtype=np.float32).tolist())
        lengths.extend(np.asarray(steps, dtype=np.float32).tolist())

    return np.asarray(scores, dtype=np.float32), np.asarray(lengths, dtype=np.float32)


def _run_two_sided_matchup(genome_right, genome_left, seed, max_steps=3000):
    batched_right = genomes_to_batched_policy([genome_right])
    batched_left = genomes_to_batched_policy([genome_left])
    score_right, steps_right = _run_match_batch_jit(
        batched_right,
        batched_left,
        seed,
        np.asarray([seed], dtype=np.uint32),
        max_steps,
    )
    score_left_perspective, steps_left = _run_match_batch_jit(
        batched_left,
        batched_right,
        seed + 1,
        np.asarray([seed + 1], dtype=np.uint32),
        max_steps,
    )
    return (
        float(np.asarray(score_right)[0] - np.asarray(score_left_perspective)[0]),
        float(np.asarray(steps_right)[0] + np.asarray(steps_left)[0]),
        2,
    )


def evaluate_selfplay_population(
    population,
    opponents_per_genome,
    seed_base,
    max_tournaments=None,
    baseline_seed_base=None,
    hall_of_fame_genomes=None,
    hall_of_fame_opponents=0,
    baseline_fitness_episodes=0,
    baseline_fitness_weight=0.0,
    max_steps=3000,
    batch_size=256,
):
    n = len(population.members)
    pairings, owners, local_indices = _build_index_pairings(
        n,
        opponents_per_genome,
        seed_base,
        max_tournaments=max_tournaments,
    )
    total_scores, total_lengths, games_played = _evaluate_pairings_from_indices(
        population,
        pairings,
        owners,
        local_indices,
        seed_base,
        max_steps,
        batch_size,
    )

    hall_of_fame_genomes = hall_of_fame_genomes or []
    if hall_of_fame_genomes and hall_of_fame_opponents > 0:
        rng = np.random.default_rng(seed_base)
        for i, genome in enumerate(population.members):
            sample_size = min(hall_of_fame_opponents, len(hall_of_fame_genomes))
            hall_indices = rng.choice(len(hall_of_fame_genomes), size=sample_size, replace=False)
            for local_idx, hall_idx in enumerate(hall_indices):
                score, steps, games = _run_two_sided_matchup(
                    genome,
                    hall_of_fame_genomes[int(hall_idx)],
                    seed_base + 500000 + i * 1000 + local_idx * 10,
                    max_steps=max_steps,
                )
                total_scores[i] += score
                total_lengths[i] += steps
                games_played[i] += games

    tournament_baseline_scores = []
    if baseline_seed_base is not None and pairings.shape[0] > 0:
        baseline_genomes = [population.members[int(owner)] for owner in owners]
        baseline_seeds = [baseline_seed_base + idx for idx in range(pairings.shape[0])]
        baseline_scores, _baseline_lengths = _evaluate_genome_vs_baseline_batch(
            baseline_genomes,
            baseline_seeds,
            max_steps=max_steps,
            batch_size=batch_size,
        )
        tournament_baseline_scores = [float(x) for x in baseline_scores]

    selfplay_fitness_weight = max(0.0, 1.0 - baseline_fitness_weight)
    avg_scores = []
    avg_lengths = []
    for i, genome in enumerate(population.members):
        divisor = max(1, int(games_played[i]))
        avg_score = float(total_scores[i] / divisor)
        avg_length = float(total_lengths[i] / divisor)
        avg_scores.append(avg_score)
        avg_lengths.append(avg_length)
        fitness_score = avg_score

        if baseline_fitness_episodes > 0 and baseline_fitness_weight > 0.0:
            baseline_genomes = [genome] * baseline_fitness_episodes
            baseline_seeds = [
                seed_base + 800000 + i * 100 + episode_idx
                for episode_idx in range(baseline_fitness_episodes)
            ]
            baseline_scores, _baseline_lengths = _evaluate_genome_vs_baseline_batch(
                baseline_genomes,
                baseline_seeds,
                max_steps=max_steps,
                batch_size=batch_size,
            )
            baseline_mean = float(np.mean(baseline_scores)) if len(baseline_scores) else 0.0
            fitness_score = (
                selfplay_fitness_weight * avg_score
                + baseline_fitness_weight * baseline_mean
            )

        genome.fitness = fitness_score + 6.0

    return avg_scores, avg_lengths, tournament_baseline_scores


def evaluate_selfplay_population_gpu(*args, **kwargs):
    return evaluate_selfplay_population(*args, **kwargs)


def evaluate_vs_baseline(genome, episodes, seed_base, max_steps=3000, batch_size=256):
    genomes = [genome] * episodes
    seeds = [seed_base + i for i in range(episodes)]
    scores, lengths = _evaluate_genome_vs_baseline_batch(
        genomes,
        seeds,
        max_steps=max_steps,
        batch_size=batch_size,
    )
    if len(scores) == 0:
        return 0.0, 0.0, 0.0, []
    return (
        float(np.mean(scores)),
        float(np.std(scores)),
        float(np.mean(lengths)),
        [float(x) for x in scores],
    )
