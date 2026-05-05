from __future__ import annotations

from dataclasses import dataclass

from slimevolley.verify.policies import make_policy
from slimevolley.verify.state_trace import gym_trace_row


@dataclass(frozen=True)
class RolloutResult:
    backend: str
    matchup: str
    seed: int
    score: float
    steps: int
    right_life: int
    left_life: int


def _matchup_policies(matchup: str, seed: int, genome_path=None):
    if matchup == "baseline_vs_baseline":
        return make_policy("baseline", seed), make_policy("baseline", seed + 1)
    if matchup == "random_vs_baseline":
        return make_policy("random", seed), make_policy("baseline", seed + 1)
    if matchup == "noop_vs_baseline":
        return make_policy("noop", seed), make_policy("baseline", seed + 1)
    if matchup == "genome_vs_baseline":
        return make_policy("genome", seed, genome_path), make_policy("baseline", seed + 1)
    raise ValueError(f"Unsupported matchup: {matchup}")


def run_gym_rollout(matchup: str, seed: int, genome_path=None, trace: bool = False, max_steps: int = 3000):
    import gym
    import slimevolleygym  # noqa: F401

    right_policy, left_policy = _matchup_policies(matchup, seed, genome_path)
    if hasattr(right_policy, "reset"):
        right_policy.reset()
    if hasattr(left_policy, "reset"):
        left_policy.reset()

    env = gym.make("SlimeVolley-v0").unwrapped
    env.seed(seed)
    obs_right = env.reset()
    obs_left = env.game.agent_left.getObservation()

    done = False
    score = 0.0
    steps = 0
    rows = []

    while not done and steps < max_steps:
        action_right = right_policy.predict(obs_right)
        action_left = left_policy.predict(obs_left)
        obs_right, reward, done, info = env.step(action_right, action_left)
        obs_left = info["otherObs"]
        score += float(reward)
        steps += 1
        if trace:
            rows.append(gym_trace_row(env, steps, float(reward), done))

    result = RolloutResult(
        backend="gym",
        matchup=matchup,
        seed=seed,
        score=float(score),
        steps=int(steps),
        right_life=int(env.game.agent_right.life),
        left_life=int(env.game.agent_left.life),
    )
    env.close()
    return result, rows

