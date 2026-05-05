from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ACTION_TABLE = np.asarray(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 0, 1],
        [0, 0, 1],
        [0, 1, 1],
        [0, 1, 0],
    ],
    dtype=np.int8,
)


@dataclass
class NoopPolicy:
    def reset(self) -> None:
        pass

    def predict(self, obs):
        return [0, 0, 0]


@dataclass
class RandomPolicy:
    seed: int

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def predict(self, obs):
        idx = int(self.rng.integers(0, len(ACTION_TABLE)))
        return ACTION_TABLE[idx].astype(int).tolist()


@dataclass
class GenomePolicy:
    genome: object

    def reset(self) -> None:
        pass

    def predict(self, obs):
        from slimevolley.selfplay.policy import genome_to_action

        return genome_to_action(self.genome, obs)


def load_genome(path: str | Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def make_gym_baseline_policy():
    import slimevolleygym

    policy = slimevolleygym.BaselinePolicy()
    policy.reset()
    return policy


def make_policy(name: str, seed: int = 0, genome_path: str | Path | None = None):
    if name == "noop":
        return NoopPolicy()
    if name == "random":
        return RandomPolicy(seed=seed)
    if name == "baseline":
        return make_gym_baseline_policy()
    if name == "genome":
        if genome_path is None:
            raise ValueError("--genome is required for genome policies.")
        return GenomePolicy(load_genome(genome_path))
    raise ValueError(f"Unsupported policy: {name}")

