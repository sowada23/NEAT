
"""
Evaluate how well genomes perform.

This file does not train directly.
Instead, it measures performance by running many episodes and
computing average scores.

It contains two kinds of evaluation:
- self-play evaluation across the current population
- benchmark evaluation against the fixed baseline policy

In short:
- chooses opponents
- runs matches
- averages scores and episode lengths
- assigns fitness values to genomes
"""

import numpy as np

from .episodes import run_selfplay_episode, run_vs_baseline_episode


def evaluate_selfplay_population(population, opponents_per_genome, seed_base):
    """
    Evaluate each genome only against other genomes in the current population.

    Each sampled pairing plays two matches with swapped sides to reduce side bias.
    """
    n = len(population.members)
    total_scores = [0.0] * n
    total_lengths = [0.0] * n
    games_played = [0] * n

    rng = np.random.default_rng(seed_base)

    for i, genome in enumerate(population.members):
        candidate_indices = [j for j in range(n) if j != i]
        if not candidate_indices:
            continue

        sample_size = min(opponents_per_genome, len(candidate_indices))
        opponents = rng.choice(candidate_indices, size=sample_size, replace=False)

        for local_idx, j in enumerate(opponents):
            opponent = population.members[int(j)]
            match_seed = seed_base + i * 1000 + local_idx * 10

            score_right, steps_right, _ = run_selfplay_episode(
                genome_right=genome,
                genome_left=opponent,
                seed=match_seed,
                capture_frames=False,
            )
            score_left_perspective, steps_left, _ = run_selfplay_episode(
                genome_right=opponent,
                genome_left=genome,
                seed=match_seed + 1,
                capture_frames=False,
            )

            genome_score_left = -score_left_perspective

            total_scores[i] += score_right + genome_score_left
            total_lengths[i] += steps_right + steps_left
            games_played[i] += 2

    avg_scores = []
    avg_lengths = []
    for i, genome in enumerate(population.members):
        divisor = max(1, games_played[i])
        avg_score = total_scores[i] / divisor
        avg_length = total_lengths[i] / divisor
        avg_scores.append(avg_score)
        avg_lengths.append(avg_length)
        genome.fitness = avg_score + 6.0

    return avg_scores, avg_lengths


def evaluate_vs_baseline(genome, episodes, seed_base):
    """Benchmark a genome against the fixed baseline policy."""
    scores = []
    lengths = []

    for i in range(episodes):
        score, steps, _ = run_vs_baseline_episode(
            genome=genome,
            seed=seed_base + i,
            capture_frames=False,
        )
        scores.append(score)
        lengths.append(steps)

    return (
        float(np.mean(scores)),
        float(np.std(scores)),
        float(np.mean(lengths)),
        scores,
    )
