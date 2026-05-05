from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slimevolley.baseline_train.episodes import run_vs_baseline_episode


@dataclass(frozen=True)
class GenomeMetrics:
    fitness: float
    score_mean: float
    score_std: float
    steps_mean: float
    right_life_mean: float
    left_life_mean: float
    node_count: int
    conn_count: int
    enabled_conn_count: int


def genome_complexity(genome) -> tuple[int, int, int]:
    node_count = len(genome.nodes)
    conn_count = len(genome.connections)
    enabled_conn_count = sum(1 for conn in genome.connections.values() if conn.enabled)
    return node_count, conn_count, enabled_conn_count


def evaluate_genome_vs_baseline(
    genome,
    episodes: int,
    seed_base: int,
    threshold: float = 0.0,
    max_steps: int = 3000,
    survival_bonus_weight: float = 0.0,
    complexity_penalty_weight: float = 0.0,
) -> GenomeMetrics:
    results = [
        run_vs_baseline_episode(
            genome,
            seed=seed_base + episode_idx,
            threshold=threshold,
            max_steps=max_steps,
        )
        for episode_idx in range(episodes)
    ]
    scores = np.asarray([r.score for r in results], dtype=np.float32)
    steps = np.asarray([r.steps for r in results], dtype=np.float32)
    right_lives = np.asarray([r.right_life for r in results], dtype=np.float32)
    left_lives = np.asarray([r.left_life for r in results], dtype=np.float32)

    node_count, conn_count, enabled_conn_count = genome_complexity(genome)
    score_mean = float(scores.mean()) if len(scores) else 0.0
    score_std = float(scores.std()) if len(scores) else 0.0
    steps_mean = float(steps.mean()) if len(steps) else 0.0
    survival_bonus = survival_bonus_weight * (steps_mean / max(1.0, float(max_steps)))
    complexity_penalty = complexity_penalty_weight * float(node_count + enabled_conn_count)
    raw_fitness = score_mean + survival_bonus - complexity_penalty
    fitness = raw_fitness + 6.0

    return GenomeMetrics(
        fitness=float(fitness),
        score_mean=score_mean,
        score_std=score_std,
        steps_mean=steps_mean,
        right_life_mean=float(right_lives.mean()) if len(right_lives) else 0.0,
        left_life_mean=float(left_lives.mean()) if len(left_lives) else 0.0,
        node_count=node_count,
        conn_count=conn_count,
        enabled_conn_count=enabled_conn_count,
    )


def evaluate_population_vs_baseline(
    population,
    episodes_per_genome: int,
    seed_base: int,
    threshold: float = 0.0,
    max_steps: int = 3000,
    survival_bonus_weight: float = 0.0,
    complexity_penalty_weight: float = 0.0,
) -> list[GenomeMetrics]:
    metrics = []
    for genome_idx, genome in enumerate(population.members):
        genome_metrics = evaluate_genome_vs_baseline(
            genome,
            episodes=episodes_per_genome,
            seed_base=seed_base + genome_idx * 1000,
            threshold=threshold,
            max_steps=max_steps,
            survival_bonus_weight=survival_bonus_weight,
            complexity_penalty_weight=complexity_penalty_weight,
        )
        genome.fitness = genome_metrics.fitness
        metrics.append(genome_metrics)
    return metrics

