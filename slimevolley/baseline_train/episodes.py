from __future__ import annotations

from dataclasses import dataclass

from slimevolley.baseline_train.policy import genome_to_action


@dataclass(frozen=True)
class EpisodeResult:
    score: float
    steps: int
    right_life: int
    left_life: int


def run_vs_baseline_episode(genome, seed: int, threshold: float = 0.0, max_steps: int = 3000) -> EpisodeResult:
    """Run one original-Gym episode with the genome on the right vs built-in baseline on the left."""
    import gym
    import slimevolleygym  # noqa: F401

    env = gym.make("SlimeVolley-v0").unwrapped
    env.seed(seed)
    obs = env.reset()

    done = False
    total_reward = 0.0
    steps = 0

    while not done and steps < max_steps:
        action = genome_to_action(genome, obs, threshold=threshold)
        obs, reward, done, _info = env.step(action)
        total_reward += float(reward)
        steps += 1

    result = EpisodeResult(
        score=float(total_reward),
        steps=int(steps),
        right_life=int(env.game.agent_right.life),
        left_life=int(env.game.agent_left.life),
    )
    env.close()
    return result

