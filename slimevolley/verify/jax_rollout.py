from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slimevolley.verify.policies import ACTION_TABLE, load_genome
from slimevolley.verify.state_trace import jax_trace_row


@dataclass(frozen=True)
class RolloutResult:
    backend: str
    matchup: str
    seed: int
    score: float
    steps: int
    right_life: int
    left_life: int


def _single_genome_action(genome, obs):
    from slimevolley.gpu_selfplay.padded_policy import genomes_to_batched_policy, policy_actions_batched

    batched = genomes_to_batched_policy([genome])
    return np.asarray(policy_actions_batched(batched, np.asarray(obs, dtype=np.float32)[None, :]))[0]


def _policy_action(policy_name, obs, state, rng, genome=None):
    if policy_name == "noop":
        return np.asarray([0, 0, 0], dtype=np.int8), state
    if policy_name == "random":
        idx = int(rng.integers(0, len(ACTION_TABLE)))
        return ACTION_TABLE[idx], state
    if policy_name == "baseline":
        from slimevolley.gpu_selfplay.policy import baseline_actions_batched

        actions, next_state = baseline_actions_batched(np.asarray(obs, dtype=np.float32)[None, :], state)
        return np.asarray(actions)[0], next_state
    if policy_name == "genome":
        return _single_genome_action(genome, obs), state
    raise ValueError(f"Unsupported JAX policy: {policy_name}")


def _matchup_policy_names(matchup: str):
    if matchup == "baseline_vs_baseline":
        return "baseline", "baseline"
    if matchup == "random_vs_baseline":
        return "random", "baseline"
    if matchup == "noop_vs_baseline":
        return "noop", "baseline"
    if matchup == "genome_vs_baseline":
        return "genome", "baseline"
    raise ValueError(f"Unsupported matchup: {matchup}")


def run_jax_rollout(matchup: str, seed: int, genome_path=None, trace: bool = False, max_steps: int = 3000):
    import jax
    import jax.numpy as jnp

    from slimevolley.gpu_selfplay.env import batched_observations, reset_batched_env, step_batched_env
    from slimevolley.gpu_selfplay.policy import reset_baseline_state

    right_name, left_name = _matchup_policy_names(matchup)
    genome = load_genome(genome_path) if right_name == "genome" else None
    rng = np.random.default_rng(seed)

    key = jax.random.PRNGKey(seed)
    state = reset_batched_env(key[None, :])
    obs_right, obs_left = batched_observations(state)
    right_baseline_state = reset_baseline_state(1) if right_name == "baseline" else None
    left_baseline_state = reset_baseline_state(1) if left_name == "baseline" else None

    done = False
    score = 0.0
    steps = 0
    rows = []

    while not done and steps < max_steps:
        action_right, right_baseline_state = _policy_action(
            right_name,
            np.asarray(obs_right)[0],
            right_baseline_state,
            rng,
            genome=genome,
        )
        action_left, left_baseline_state = _policy_action(
            left_name,
            np.asarray(obs_left)[0],
            left_baseline_state,
            rng,
        )
        state, obs_right, obs_left, reward, done_array = step_batched_env(
            state,
            jnp.asarray(action_right[None, :]),
            jnp.asarray(action_left[None, :]),
            max_steps=max_steps,
        )
        reward_value = float(np.asarray(reward)[0])
        done = bool(np.asarray(done_array)[0])
        score += reward_value
        steps += 1
        if trace:
            rows.append(jax_trace_row(state, steps, reward_value, done))

    result = RolloutResult(
        backend="jax",
        matchup=matchup,
        seed=seed,
        score=float(score),
        steps=int(steps),
        right_life=int(np.asarray(state.agent_right.life)[0]),
        left_life=int(np.asarray(state.agent_left.life)[0]),
    )
    return result, rows

