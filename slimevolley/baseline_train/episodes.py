from __future__ import annotations

from dataclasses import dataclass

from slimevolley.baseline_train.policy import genome_action_with_stats


@dataclass(frozen=True)
class EpisodeResult:
    score: float
    steps: int
    right_life: int
    left_life: int
    action_counts: dict[str, int]
    raw_action_counts: dict[str, int]
    conflict_count: int


ACTION_COUNT_KEYS = (
    "noop",
    "forward",
    "backward",
    "jump",
    "forward_jump",
    "backward_jump",
    "conflict",
    "conflict_jump",
)


def _empty_action_counts() -> dict[str, int]:
    return {key: 0 for key in ACTION_COUNT_KEYS}


def run_vs_baseline_episode(
    genome,
    seed: int,
    threshold: float = 0.0,
    max_steps: int = 3000,
    resolve_conflict: bool = True,
) -> EpisodeResult:
    """Run one original-Gym episode with the genome on the right vs built-in baseline on the left."""
    import gym
    import slimevolleygym  # noqa: F401

    env = gym.make("SlimeVolley-v0").unwrapped
    env.seed(seed)
    obs = env.reset()

    done = False
    total_reward = 0.0
    steps = 0
    action_counts = _empty_action_counts()
    raw_action_counts = _empty_action_counts()
    conflict_count = 0

    while not done and steps < max_steps:
        action, raw_action_name, action_name = genome_action_with_stats(
            genome,
            obs,
            threshold=threshold,
            resolve_conflict=resolve_conflict,
        )
        raw_action_counts[raw_action_name] += 1
        action_counts[action_name] += 1
        if raw_action_name in ("conflict", "conflict_jump"):
            conflict_count += 1
        obs, reward, done, _info = env.step(action)
        total_reward += float(reward)
        steps += 1

    result = EpisodeResult(
        score=float(total_reward),
        steps=int(steps),
        right_life=int(env.game.agent_right.life),
        left_life=int(env.game.agent_left.life),
        action_counts=action_counts,
        raw_action_counts=raw_action_counts,
        conflict_count=conflict_count,
    )
    env.close()
    return result
